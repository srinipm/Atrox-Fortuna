/*
=============================================
MongoDB Conversion Example
This shows how a converted Sybase stored procedure might look after conversion
=============================================
*/

// Original Sybase stored procedure:
/*
CREATE PROCEDURE get_customer_orders @customer_id INT
AS
BEGIN
    SELECT 
        c.customer_name,
        o.order_id,
        o.order_date,
        o.total_amount
    FROM 
        customers c
        JOIN orders o ON c.customer_id = o.customer_id
    WHERE 
        c.customer_id = @customer_id
    ORDER BY 
        o.order_date DESC
END
*/

// Converted MongoDB function:
function get_customer_orders(customer_id) {
    // MongoDB aggregation pipeline to join customers and orders
    return db.customers.aggregate([
        {
            $match: {
                customer_id: customer_id /* Converted WHERE condition */
            }
        },
        {
            $lookup: {
                from: "orders",
                localField: "customer_id",
                foreignField: "customer_id",
                as: "customer_orders"
            } /* Suggested MongoDB $lookup for JOIN operation */
        },
        {
            $unwind: "$customer_orders" /* Flatten the array created by $lookup */
        },
        {
            $project: {
                customer_name: 1,
                order_id: "$customer_orders.order_id",
                order_date: "$customer_orders.order_date",
                total_amount: "$customer_orders.total_amount"
            } /* Projection of selected fields */
        },
        {
            $sort: {
                "customer_orders.order_date": -1
            } /* Sort by order_date DESC */
        }
    ]).toArray(); /* Convert cursor to array for immediate results */
}

// Example usage:
// const orders = get_customer_orders(12345);
// console.log(orders);

// For transaction support:
function create_order_with_items(customer_id, order_details, items) {
    const session = db.getMongo().startSession();
    session.startTransaction();
    
    try {
        // Insert the order document
        const orderResult = db.orders.insertOne({
            customer_id: customer_id,
            order_date: new Date(),
            total_amount: order_details.total_amount,
            status: "new"
        }, { session });
        
        const order_id = orderResult.insertedId;
        
        // Insert all order items
        const orderItems = items.map(item => ({
            order_id: order_id,
            product_id: item.product_id,
            quantity: item.quantity,
            price: item.price
        }));
        
        db.order_items.insertMany(orderItems, { session });
        
        // Update inventory (example of another operation in the same transaction)
        for (const item of items) {
            db.inventory.updateOne(
                { product_id: item.product_id },
                { $inc: { stock_level: -item.quantity } },
                { session }
            );
        }
        
        // Commit the transaction
        session.commitTransaction();
        return order_id;
    } catch (error) {
        // Abort on error
        session.abortTransaction();
        throw error;
    } finally {
        session.endSession();
    }
}
