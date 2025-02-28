/*
** ==========================================================================
** Complex Order Management and Analytics Procedure for Sybase ASE
** --------------------------------------------------------------------------
**
** Author:        Senior Sybase DBA
** Creation Date: 2024-10-30
** Version:       1.0
**
** Description:
**   Enterprise-grade stored procedure for order management system with:
**    - Order processing with validation
**    - Inventory management with auto-replenishment
**    - Multi-currency support and exchange rate handling
**    - Customer analytics and segmentation
**    - Promotional code validation and application
**    - Comprehensive audit logging and error handling
**    - Dynamic SQL for performance optimization
**
** ==========================================================================
*/

IF EXISTS (SELECT 1 FROM sysobjects WHERE name = 'sp_order_management_system' AND type = 'P')
    DROP PROCEDURE sp_order_management_system
GO

CREATE PROCEDURE sp_order_management_system
    @p_operation_type           VARCHAR(50),               -- Operation type: 'NEW_ORDER', 'UPDATE_ORDER', 'CANCEL_ORDER', 'ANALYTICS', etc.
    @p_customer_id              INT            = NULL,     -- Customer ID
    @p_order_id                 INT            = NULL,     -- Order ID for existing orders
    @p_order_date               DATETIME       = NULL,     -- Order date (defaults to current date if null)
    @p_order_items              TEXT           = NULL,     -- JSON-formatted order items
    @p_payment_method           VARCHAR(30)    = NULL,     -- Payment method
    @p_currency                 CHAR(3)        = 'USD',    -- Currency code (ISO)
    @p_shipping_address         VARCHAR(500)   = NULL,     -- Shipping address
    @p_promo_code               VARCHAR(30)    = NULL,     -- Promotional code
    @p_analytics_type           VARCHAR(50)    = NULL,     -- Type of analytics to run
    @p_date_range_start         DATETIME       = NULL,     -- Start date for analytics
    @p_date_range_end           DATETIME       = NULL,     -- End date for analytics
    @p_segment_criteria         VARCHAR(100)   = NULL,     -- Customer segmentation criteria
    @p_batch_size               INT            = 1000,     -- Batch size for processing
    @p_log_level                TINYINT        = 3,        -- Log level (1=ERROR, 2=WARN, 3=INFO, 4=DEBUG)
    @p_user_role                VARCHAR(30)    = NULL,     -- User role for permission checks
    @p_transaction_isolation    VARCHAR(20)    = 'READ COMMITTED'  -- Transaction isolation level
AS
BEGIN
    -- Local variable declarations - grouped by functional area
    -- System and control variables
    DECLARE @v_error_code INT, @v_error_message VARCHAR(4000), @v_transaction_started BIT
    DECLARE @v_current_date DATETIME, @v_proc_name VARCHAR(255), @v_step_id INT
    DECLARE @v_step_desc VARCHAR(500), @v_row_count INT, @v_process_id INT
    DECLARE @v_start_time DATETIME, @v_end_time DATETIME, @v_execution_time_ms INT
    DECLARE @v_system_user VARCHAR(50), @v_db_name VARCHAR(50), @v_server_name VARCHAR(50)
    DECLARE @v_api_version VARCHAR(10), @v_debug_mode BIT, @v_permission_check BIT
    DECLARE @v_sql VARCHAR(MAX), @v_sql_params NVARCHAR(4000), @v_temp_table_name VARCHAR(100)
    
    -- Customer-related variables
    DECLARE @v_customer_name VARCHAR(100), @v_customer_email VARCHAR(100)
    DECLARE @v_customer_status VARCHAR(20), @v_customer_segment VARCHAR(50)
    DECLARE @v_customer_lifetime_value DECIMAL(18,2)
    
    -- Order-related variables
    DECLARE @v_order_status VARCHAR(30), @v_order_subtotal DECIMAL(18,2)
    DECLARE @v_order_tax DECIMAL(18,2), @v_order_shipping DECIMAL(18,2)
    DECLARE @v_order_discount DECIMAL(18,2), @v_order_total DECIMAL(18,2)
    DECLARE @v_order_currency CHAR(3), @v_exchange_rate DECIMAL(18,6)
    DECLARE @v_order_total_usd DECIMAL(18,2)
    
    -- Item-related variables
    DECLARE @v_item_id INT, @v_item_sku VARCHAR(50), @v_item_name VARCHAR(200)
    DECLARE @v_item_quantity INT, @v_item_unit_price DECIMAL(18,2)
    DECLARE @v_item_subtotal DECIMAL(18,2), @v_item_tax_rate DECIMAL(6,4)
    DECLARE @v_item_tax DECIMAL(18,2), @v_item_total DECIMAL(18,2)
    DECLARE @v_item_cost DECIMAL(18,2), @v_item_margin DECIMAL(18,2)
    
    -- Inventory-related variables
    DECLARE @v_inventory_id INT, @v_inventory_quantity INT
    DECLARE @v_inventory_threshold INT, @v_inventory_reorder_qty INT
    
    -- Payment-related variables
    DECLARE @v_payment_status VARCHAR(30), @v_payment_confirmation VARCHAR(100)
    
    -- Promo-related variables
    DECLARE @v_promo_discount_pct DECIMAL(6,4), @v_promo_discount_amt DECIMAL(18,2)
    DECLARE @v_promo_expiry_date DATETIME, @v_promo_usage_limit INT
    DECLARE @v_promo_usage_count INT, @v_promo_is_valid BIT
    
    -- Analytics-related variables
    DECLARE @v_analytics_metric VARCHAR(100), @v_analytics_value DECIMAL(18,4)
    DECLARE @v_analytics_trend DECIMAL(9,6), @v_analytics_period VARCHAR(20)
    
    -- Cursor declarations
    DECLARE @items_cursor CURSOR
    DECLARE @analytics_cursor CURSOR
    DECLARE @inventory_cursor CURSOR
    
    -- Initialize procedure variables
    SET @v_proc_name = 'sp_order_management_system'
    SET @v_transaction_started = 0
    SET @v_current_date = GETDATE()
    SET @v_step_id = 0
    SET @v_system_user = SYSTEM_USER
    SET @v_db_name = DB_NAME()
    SET @v_server_name = @@SERVERNAME
    SET @v_start_time = GETDATE()
    SET @v_debug_mode = CASE WHEN @p_log_level >= 4 THEN 1 ELSE 0 END
    
    -- Set the isolation level for the transaction
    EXECUTE sp_lock_timeout 30000  -- 30 seconds lock timeout
    
    IF @p_transaction_isolation = 'READ UNCOMMITTED'
        SET TRANSACTION ISOLATION LEVEL 0
    ELSE IF @p_transaction_isolation = 'READ COMMITTED'
        SET TRANSACTION ISOLATION LEVEL 1
    ELSE IF @p_transaction_isolation = 'REPEATABLE READ'
        SET TRANSACTION ISOLATION LEVEL 2
    ELSE IF @p_transaction_isolation = 'SERIALIZABLE'
        SET TRANSACTION ISOLATION LEVEL 3
        
    -- Begin error handling with try-catch mechanism
    BEGIN TRY
        -- Initialize error variables and log execution start
        SET @v_error_code = 0
        SET @v_error_message = NULL
        SET @v_step_id = 1
        SET @v_step_desc = 'Procedure execution started'
        EXEC sp_log_message @p_log_level, @v_proc_name, @v_step_id, @v_step_desc, 'INFO'
        
        -- Parameter validation
        SET @v_step_id = 2
        SET @v_step_desc = 'Validating parameters'
        
        IF @p_operation_type IS NULL
            RAISERROR('Operation type parameter is required', 16, 1)
        
        -- Operation-specific parameter validation
        IF @p_operation_type = 'NEW_ORDER'
        BEGIN
            IF @p_customer_id IS NULL OR @p_order_items IS NULL OR @p_payment_method IS NULL
                RAISERROR('Customer ID, order items, and payment method are required for new orders', 16, 1)
                
            IF @p_order_date IS NULL
                SET @p_order_date = @v_current_date
        END
        ELSE IF @p_operation_type IN ('UPDATE_ORDER', 'CANCEL_ORDER')
        BEGIN
            IF @p_order_id IS NULL
                RAISERROR('Order ID is required for update/cancel operations', 16, 1)
        END
        ELSE IF @p_operation_type = 'ANALYTICS'
        BEGIN
            IF @p_analytics_type IS NULL
                RAISERROR('Analytics type is required for analytics operation', 16, 1)
                
            IF @p_date_range_start IS NULL
                SET @p_date_range_start = DATEADD(MONTH, -1, @v_current_date)
                
            IF @p_date_range_end IS NULL
                SET @p_date_range_end = @v_current_date
        END
        
        -- Verify user permissions
        SET @v_step_id = 3
        SET @v_step_desc = 'Checking user permissions'
        
        SELECT @v_permission_check = dbo.fn_check_user_permission(@v_system_user, @p_user_role, @p_operation_type)
        
        IF @v_permission_check = 0
            RAISERROR('User does not have permission to perform this operation', 16, 1)
            
        EXEC sp_log_message @p_log_level, @v_proc_name, @v_step_id, @v_step_desc, 'INFO'
        
        -- Begin transaction for data modification operations
        IF @p_operation_type IN ('NEW_ORDER', 'UPDATE_ORDER', 'CANCEL_ORDER', 'INVENTORY_UPDATE')
        BEGIN
            SET @v_step_id = 4
            SET @v_step_desc = 'Beginning transaction'
            
            BEGIN TRANSACTION
            SET @v_transaction_started = 1
            
            EXEC sp_log_message @p_log_level, @v_proc_name, @v_step_id, @v_step_desc, 'INFO'
        END
        
        -- Generate process ID for tracking
        SET @v_process_id = NEXT VALUE FOR seq_process_id
        
        -- Main operation switch
        SET @v_step_id = 5
        SET @v_step_desc = 'Executing operation: ' + @p_operation_type
        
        -- Execute the requested operation
        CASE @p_operation_type
            -- NEW ORDER PROCESSING
            WHEN 'NEW_ORDER' THEN
            BEGIN
                -- Get customer information and validate status
                SET @v_step_id = 10
                SET @v_step_desc = 'Retrieving customer information'
                
                SELECT 
                    @v_customer_name = c.customer_name,
                    @v_customer_email = c.email,
                    @v_customer_segment = c.segment,
                    @v_customer_lifetime_value = c.lifetime_value,
                    @v_customer_status = c.status
                FROM 
                    customers c
                WHERE 
                    c.customer_id = @p_customer_id
                    
                IF @@ROWCOUNT = 0 OR @v_customer_status <> 'ACTIVE'
                    RAISERROR('Customer not found or inactive', 16, 1)
                
                EXEC sp_log_message @p_log_level, @v_proc_name, @v_step_id, @v_step_desc, 'INFO'
                
                -- Initialize order data
                SET @v_step_id = 11
                SET @v_step_desc = 'Initializing order data'
                SET @v_order_status = 'PENDING'
                
                -- Currency conversion if needed
                IF @p_currency <> 'USD'
                BEGIN
                    SELECT @v_exchange_rate = exchange_rate 
                    FROM currency_rates 
                    WHERE currency_code = @p_currency
                      AND effective_date = (
                          SELECT MAX(effective_date) 
                          FROM currency_rates 
                          WHERE currency_code = @p_currency
                            AND effective_date <= @v_current_date
                      )
                      
                    IF @v_exchange_rate IS NULL
                        RAISERROR('Exchange rate not found for specified currency', 16, 1)
                END
                ELSE
                    SET @v_exchange_rate = 1.0
                
                -- Initialize order totals
                SET @v_order_subtotal = 0
                SET @v_order_tax = 0
                SET @v_order_shipping = 0
                SET @v_order_discount = 0
                
                -- Validate promotional code if provided
                IF @p_promo_code IS NOT NULL
                BEGIN
                    SET @v_step_id = 12
                    SET @v_step_desc = 'Validating promotional code'
                    
                    SELECT 
                        @v_promo_discount_pct = discount_percentage,
                        @v_promo_discount_amt = discount_amount,
                        @v_promo_expiry_date = expiry_date,
                        @v_promo_usage_limit = usage_limit,
                        @v_promo_usage_count = usage_count,
                        @v_promo_is_valid = is_active
                    FROM 
                        promotional_codes
                    WHERE 
                        code = @p_promo_code
                        
                    IF @@ROWCOUNT = 0 OR @v_promo_is_valid = 0 OR 
                       @v_promo_expiry_date < @v_current_date OR
                       (@v_promo_usage_limit IS NOT NULL AND @v_promo_usage_count >= @v_promo_usage_limit)
                        RAISERROR('Invalid or expired promotional code', 16, 1)
                        
                    -- Increment usage count
                    UPDATE promotional_codes
                    SET usage_count = usage_count + 1
                    WHERE code = @p_promo_code
                END
                
                -- Parse JSON order items and create temp table
                SET @v_step_id = 13
                SET @v_step_desc = 'Processing order items'
                
                CREATE TABLE #temp_order_items (
                    item_id INT,
                    item_sku VARCHAR(50),
                    quantity INT,
                    unit_price DECIMAL(18,2),
                    tax_rate DECIMAL(6,4)
                )
                
                -- Parse JSON and insert into temp table using Sybase's JSON parsing function
                INSERT INTO #temp_order_items
                EXEC sp_parse_json_order_items @p_order_items
                
                -- Process each order item
                SET @v_step_id = 14
                SET @v_step_desc = 'Processing individual order items'
                
                DECLARE @items_cursor CURSOR FOR
                SELECT item_id, item_sku, quantity, unit_price, tax_rate
                FROM #temp_order_items
                
                OPEN @items_cursor
                FETCH NEXT FROM @items_cursor INTO @v_item_id, @v_item_sku, @v_item_quantity, @v_item_unit_price, @v_item_tax_rate
                
                WHILE @@FETCH_STATUS = 0
                BEGIN
                    -- Check inventory and product details
                    SELECT 
                        @v_item_name = p.product_name,
                        @v_item_cost = p.cost,
                        @v_inventory_quantity = i.quantity,
                        @v_inventory_threshold = i.reorder_threshold,
                        @v_inventory_id = i.inventory_id
                    FROM 
                        products p
                    JOIN 
                        inventory i ON p.product_id = i.product_id
                    WHERE 
                        p.sku = @v_item_sku
                        
                    IF @@ROWCOUNT = 0
                        RAISERROR('Product not found: %s', 16, 1, @v_item_sku)
                        
                    -- Check inventory availability
                    IF @v_inventory_quantity < @v_item_quantity
                        RAISERROR('Insufficient inventory for product: %s', 16, 1, @v_item_sku)
                    
                    -- Calculate item totals and update order totals
                    SET @v_item_subtotal = @v_item_quantity * @v_item_unit_price
                    SET @v_item_tax = @v_item_subtotal * @v_item_tax_rate
                    SET @v_order_subtotal = @v_order_subtotal + @v_item_subtotal
                    SET @v_order_tax = @v_order_tax + @v_item_tax
                    
                    -- Update inventory
                    UPDATE inventory
                    SET quantity = quantity - @v_item_quantity,
                        last_updated = @v_current_date
                    WHERE inventory_id = @v_inventory_id
                    
                    -- Trigger inventory reorder if needed
                    IF (@v_inventory_quantity - @v_item_quantity) <= @v_inventory_threshold
                    BEGIN
                        -- Calculate reorder quantity based on usage pattern and lead time
                        SET @v_inventory_reorder_qty = dbo.fn_calculate_reorder_quantity(
                            @v_inventory_id, 
                            @v_current_date, 
                            DATEADD(MONTH, -3, @v_current_date)
                        )
                        
                        -- Create reorder record
                        INSERT INTO inventory_reorders (
                            inventory_id, reorder_date, quantity, status, created_by, created_date
                        )
                        VALUES (
                            @v_inventory_id, @v_current_date, @v_inventory_reorder_qty, 
                            'PENDING', @v_system_user, @v_current_date
                        )
                        
                        -- Log reorder event
                        EXEC sp_log_message @p_log_level, @v_proc_name, @v_step_id, 
                            'Inventory reorder triggered for: ' + @v_item_sku, 'WARN'
                    END
                    
                    FETCH NEXT FROM @items_cursor INTO @v_item_id, @v_item_sku, @v_item_quantity, @v_item_unit_price, @v_item_tax_rate
                END
                
                CLOSE @items_cursor
                DEALLOCATE @items_cursor
                
                -- Calculate shipping and discounts
                SET @v_step_id = 15
                SET @v_step_desc = 'Finalizing order calculations'
                
                -- Call shipping calculation procedure
                EXEC sp_calculate_shipping 
                    @p_order_items, 
                    @p_shipping_address, 
                    @v_order_shipping OUTPUT
                
                -- Apply promotional discount if applicable
                IF @p_promo_code IS NOT NULL
                BEGIN
                    IF @v_promo_discount_pct > 0
                        SET @v_order_discount = @v_order_subtotal * @v_promo_discount_pct
                    ELSE
                        SET @v_order_discount = @v_promo_discount_amt
                END
                
                -- Calculate final order total
                SET @v_order_total = @v_order_subtotal + @v_order_tax + @v_order_shipping - @v_order_discount
                SET @v_order_total_usd = @v_order_total / @v_exchange_rate
                
                -- Create order record
                SET @v_step_id = 16
                SET @v_step_desc = 'Creating order record'
                
                INSERT INTO orders (
                    customer_id, order_date, status, subtotal, tax_amount,
                    shipping_amount, discount_amount, total_amount, currency_code,
                    exchange_rate, total_amount_usd, payment_method, shipping_address,
                    promo_code, created_by, created_date
                )
                VALUES (
                    @p_customer_id, @p_order_date, @v_order_status, @v_order_subtotal, @v_order_tax,
                    @v_order_shipping, @v_order_discount, @v_order_total, @p_currency,
                    @v_exchange_rate, @v_order_total_usd, @p_payment_method, @p_shipping_address,
                    @p_promo_code, @v_system_user, @v_current_date
                )
                
                -- Get the new order ID
                SET @p_order_id = SCOPE_IDENTITY()
                
                -- Insert order items
                SET @v_step_id = 17
                SET @v_step_desc = 'Creating order detail records'
                
                INSERT INTO order_details (
                    order_id, product_id, quantity, unit_price, subtotal,
                    tax_rate, tax_amount, total_amount, created_by, created_date
                )
                SELECT 
                    @p_order_id, p.product_id, t.quantity, t.unit_price, 
                    t.quantity * t.unit_price, t.tax_rate,
                    (t.quantity * t.unit_price * t.tax_rate),
                    (t.quantity * t.unit_price) + (t.quantity * t.unit_price * t.tax_rate),
                    @v_system_user, @v_current_date
                FROM 
                    #temp_order_items t
                JOIN 
                    products p ON t.item_sku = p.sku
                
                -- Process payment
                SET @v_step_id = 18
                SET @v_step_desc = 'Processing payment'
                
                EXEC sp_process_payment 
                    @p_order_id, @p_payment_method, @v_order_total, @p_currency,
                    @v_payment_status OUTPUT, @v_payment_confirmation OUTPUT
                
                IF @v_payment_status <> 'APPROVED'
                    RAISERROR('Payment processing failed: %s', 16, 1, @v_payment_status)
                
                -- Create payment record and update order status
                INSERT INTO payments (
                    order_id, payment_method, amount, currency_code, status,
                    confirmation_code, payment_date, created_by, created_date
                )
                VALUES (
                    @p_order_id, @p_payment_method, @v_order_total, @p_currency, @v_payment_status,
                    @v_payment_confirmation, @v_current_date, @v_system_user, @v_current_date
                )
                
                -- Update order status to confirmed
                UPDATE orders
                SET status = 'CONFIRMED'
                WHERE order_id = @p_order_id
                
                -- Update customer metrics and add to order tracking
                UPDATE customers
                SET 
                    lifetime_value = lifetime_value + @v_order_total_usd,
                    last_order_date = @v_current_date,
                    order_count = order_count + 1,
                    last_modified = @v_current_date
                WHERE 
                    customer_id = @p_customer_id
                
                -- Create tracking entry
                INSERT INTO order_tracking (
                    order_id, status, status_date, comments, created_by, created_date
                )
                VALUES (
                    @p_order_id, 'ORDER_CREATED', @v_current_date, 
                    'Order created successfully', @v_system_user, @v_current_date
                )
                
                -- Return order summary
                SELECT 
                    o.order_id, o.order_date, o.status, o.total_amount, o.currency_code,
                    p.status AS payment_status, p.confirmation_code
                FROM 
                    orders o
                JOIN 
                    payments p ON o.order_id = p.order_id
                WHERE 
                    o.order_id = @p_order_id
            END
            
            -- ANALYTICS PROCESSING
            WHEN 'ANALYTICS' THEN
            BEGIN
                SET @v_step_id = 30
                SET @v_step_desc = 'Running analytics: ' + @p_analytics_type
                
                -- Create temp table for analytics results
                CREATE TABLE #analytics_results (
                    metric_name VARCHAR(100),
                    dimension VARCHAR(100),
                    period VARCHAR(50),
                    numeric_value DECIMAL(18,4),
                    trend_value DECIMAL(9,6),
                    string_value VARCHAR(MAX)
                )
                
                -- Execute appropriate analytics based on type
                IF @p_analytics_type = 'SALES_PERFORMANCE'
                BEGIN
                    -- Insert period-over-period sales comparison
                    INSERT INTO #analytics_results
                    EXEC sp_analyze_sales_performance 
                        @p_date_range_start, 
                        @p_date_range_end, 
                        @p_segment_criteria
                END
                ELSE IF @p_analytics_type = 'CUSTOMER_SEGMENTATION'
                BEGIN
                    -- Perform RFM analysis and customer segmentation
                    INSERT INTO #analytics_results
                    EXEC sp_analyze_customer_segments
                        @p_date_range_start,
                        @p_date_range_end,
                        @p_segment_criteria
                END
                ELSE IF @p_analytics_type = 'INVENTORY_ANALYSIS'
                BEGIN
                    -- Analyze inventory metrics
                    INSERT INTO #analytics_results
                    EXEC sp_analyze_inventory
                        @p_date_range_start,
                        @p_date_range_end
                END
                ELSE IF @p_analytics_type = 'PRODUCT_PERFORMANCE'
                BEGIN
                    -- Analyze product performance metrics
                    INSERT INTO #analytics_results
                    EXEC sp_analyze_product_performance
                        @p_date_range_start,
                        @p_date_range_end,
                        @p_segment_criteria
                END
                
                -- Return results to caller
                SELECT * FROM #analytics_results
                ORDER BY metric_name, dimension, period
                
                -- Log analytics execution
                EXEC sp_log_message @p_log_level, @v_proc_name, @v_step_id,
                    'Analytics completed: ' + @p_analytics_type, 'INFO'
            END
            
            -- Add other operation handlers here
            ELSE
                RAISERROR('Unsupported operation type: %s', 16, 1, @p_operation_type)
        END
        
        -- Commit transaction if one was started
        IF @v_transaction_started = 1
        BEGIN
            COMMIT TRANSACTION
            SET @v_transaction_started = 0
        END
        
        -- Calculate execution time
        SET @v_end_time = GETDATE()
        SET @v_execution_time_ms = DATEDIFF(ms, @v_start_time, @v_end_time)
        
        -- Log successful completion
        EXEC sp_log_message @p_log_level, @v_proc_name, 999,
            'Procedure completed successfully. Execution time: ' + 
            CAST(@v_execution_time_ms AS VARCHAR) + 'ms', 'INFO'
            
    END TRY
    BEGIN CATCH
        -- Error handling
        SET @v_error_code = ERROR_NUMBER()
        SET @v_error_message = ERROR_MESSAGE()
        
        -- Log the error
        EXEC sp_log_message 1, @v_proc_name, @v_step_id,
            'Error ' + CAST(@v_error_code AS VARCHAR) + ': ' + @v_error_message, 'ERROR'
        
        -- Rollback transaction if one was started
        IF @v_transaction_started = 1
        BEGIN
            ROLLBACK TRANSACTION
            SET @v_transaction_started = 0
        END
        
        -- Re-raise the error
        RAISERROR('Error in %s (Step %d): %s', 16, 1, @v_proc_name, @v_step_id, @v_error_message)
    END CATCH
END
GO

-- Create index to optimize the stored procedure
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date)
GO

-- Grant execute permissions
GRANT EXECUTE ON sp_order_management_system TO app_user, app_admin
GO