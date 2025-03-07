package com.atroxfortuna.stocklending;

import com.sybase.jdbc4.jdbc.*;
import java.sql.*;
import java.util.*;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * Global Stock Lending System - Main application class for managing stock lending operations
 * via Sybase database connection.
 * 
 * This class demonstrates best practices for working with Sybase databases in a high-volume
 * financial trading environment, including connection pooling, prepared statements,
 * transaction management, and proper resource handling.
 * 
 * @version 1.0
 */
public class GlobalStockLendingSystem implements AutoCloseable {
    private static final Logger LOGGER = Logger.getLogger(GlobalStockLendingSystem.class.getName());
    
    // Connection pool configuration
    private SybConnectionPoolDataSource dataSource;
    private int maxPoolSize = 20;
    private int minPoolSize = 5;
    private int initialPoolSize = 5;
    
    // Active connections cache
    private final Map<Connection, Boolean> activeConnections = new ConcurrentHashMap<>();
    
    // Database connection properties
    private final String serverName;
    private final int portNumber;
    private final String databaseName;
    private final String username;
    private final String password;
    
    // Query timeout (in seconds)
    private static final int QUERY_TIMEOUT = 30;
    
    // Frequently used SQL statements
    private static final String SQL_AVAILABLE_SECURITIES = 
        "SELECT s.security_id, s.isin, s.ticker, s.description, s.currency, " +
        "s.market_id, i.inventory_id, i.quantity_available, i.quantity_on_loan, " +
        "i.last_price, i.market_value " +
        "FROM securities s " +
        "INNER JOIN securities_inventory i ON s.security_id = i.security_id " +
        "WHERE i.quantity_available > 0 AND s.is_lendable = 1 " +
        "ORDER BY s.market_id, s.ticker";
    
    private static final String SQL_COUNTERPARTY_LIMITS = 
        "SELECT c.counterparty_id, c.credit_rating, cl.max_exposure, cl.current_exposure, " +
        "cl.max_single_loan, c.base_haircut, c.margin_requirement " +
        "FROM counterparties c " +
        "INNER JOIN counterparty_limits cl ON c.counterparty_id = cl.counterparty_id " +
        "WHERE c.counterparty_id = ? AND c.is_active = 1";
    
    private static final String SQL_RECORD_LOAN = 
        "INSERT INTO stock_loans (loan_ref, counterparty_id, security_id, quantity, " +
        "rate_type, loan_rate, term_days, start_date, end_date, base_currency, " +
        "market_value, collateral_id, haircut, settlement_status, trade_date, trade_time, " +
        "entered_by, settlement_location, fee_accrual_basis) " +
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    
    private static final String SQL_UPDATE_INVENTORY = 
        "UPDATE securities_inventory " +
        "SET quantity_available = quantity_available - ?, " +
        "    quantity_on_loan = quantity_on_loan + ? " +
        "WHERE security_id = ? AND quantity_available >= ?";
    
    private static final String SQL_ACTIVE_LOANS = 
        "SELECT l.loan_id, l.loan_ref, l.counterparty_id, c.counterparty_name, " +
        "l.security_id, s.ticker, s.isin, l.quantity, l.loan_rate, " +
        "l.start_date, l.end_date, l.market_value, l.settlement_status " +
        "FROM stock_loans l " +
        "INNER JOIN securities s ON l.security_id = s.security_id " +
        "INNER JOIN counterparties c ON l.counterparty_id = c.counterparty_id " +
        "WHERE l.settlement_status IN ('SETTLED', 'PENDING') " +
        "  AND l.end_date >= ?";
    
    private static final String SQL_RECALL_LOAN = 
        "UPDATE stock_loans SET recall_date = ?, recall_reason = ?, " +
        "recall_initiated_by = ?, settlement_status = 'RECALL_INITIATED' " +
        "WHERE loan_id = ? AND settlement_status IN ('SETTLED', 'PENDING')";
    
    private static final String SQL_CALCULATE_FEES = 
        "{call calculate_loan_fees(?, ?, ?)}";
    
    // Date formatter
    private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd");
    private static final DateTimeFormatter TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    /**
     * Constructs a GlobalStockLendingSystem with the specified database connection parameters.
     * 
     * @param serverName   the Sybase server hostname or IP
     * @param portNumber   the Sybase server port (usually 5000)
     * @param databaseName the name of the database
     * @param username     the database username
     * @param password     the database password
     * @throws SQLException if a database access error occurs
     */
    public GlobalStockLendingSystem(String serverName, int portNumber, String databaseName, 
                                   String username, String password) throws SQLException {
        this.serverName = serverName;
        this.portNumber = portNumber;
        this.databaseName = databaseName;
        this.username = username;
        this.password = password;
        
        initializeConnectionPool();
    }
    
    /**
     * Initializes the Sybase connection pool.
     *
     * @throws SQLException if a database access error occurs
     */
    private void initializeConnectionPool() throws SQLException {
        try {
            // Register Sybase JDBC driver
            Class.forName("com.sybase.jdbc4.jdbc.SybDriver");
            
            // Create and configure the connection pool data source
            dataSource = new SybConnectionPoolDataSource();
            dataSource.setServerName(serverName);
            dataSource.setPortNumber(portNumber);
            dataSource.setDatabaseName(databaseName);
            dataSource.setUser(username);
            dataSource.setPassword(password);
            
            // Configure connection pool properties
            dataSource.setMinPoolSize(minPoolSize);
            dataSource.setMaxPoolSize(maxPoolSize);
            dataSource.setInitialPoolSize(initialPoolSize);
            dataSource.setIdleTimeout(300);  // 5 minutes
            
            // Set additional connection properties
            Properties props = new Properties();
            props.put("APPLICATIONNAME", "GlobalStockLendingSystem");
            props.put("CHARSET", "utf8");
            props.put("LANGUAGE", "us_english");
            props.put("PACKET_SIZE", "4096");
            props.put("CONNECTION_TIMEOUT", "60"); // 1 minute
            dataSource.setProperties(props);
            
            // Initialize the connection pool by establishing initial connections
            for (int i = 0; i < initialPoolSize; i++) {
                try {
                    Connection conn = getConnection();
                    releaseConnection(conn);
                } catch (SQLException e) {
                    LOGGER.log(Level.WARNING, "Error establishing initial connection " + (i + 1), e);
                    // Continue trying to initialize remaining connections
                }
            }
            
            LOGGER.log(Level.INFO, "Sybase connection pool initialized successfully");
        } catch (ClassNotFoundException e) {
            LOGGER.log(Level.SEVERE, "Sybase JDBC driver not found", e);
            throw new SQLException("Sybase JDBC driver not found", e);
        }
    }
    
    /**
     * Gets a connection from the pool.
     *
     * @return a database connection
     * @throws SQLException if a database access error occurs
     */
    public Connection getConnection() throws SQLException {
        Connection conn = null;
        try {
            conn = dataSource.getPooledConnection().getConnection();
            
            // Set transaction isolation level to READ COMMITTED to prevent dirty reads
            // but allow non-repeatable reads and phantom reads
            conn.setTransactionIsolation(Connection.TRANSACTION_READ_COMMITTED);
            
            // Store connection in active connections map
            activeConnections.put(conn, Boolean.TRUE);
            
            LOGGER.log(Level.FINE, "Obtained database connection, active connections: {0}", activeConnections.size());
            return conn;
        } catch (SQLException e) {
            LOGGER.log(Level.SEVERE, "Error obtaining database connection", e);
            
            // If we got a connection but encountered an error while configuring it, make sure to release it
            if (conn != null) {
                try {
                    conn.close();
                } catch (SQLException ce) {
                    LOGGER.log(Level.WARNING, "Error closing connection after error", ce);
                }
            }
            
            throw e;
        }
    }
    
    /**
     * Returns a connection to the pool.
     *
     * @param conn the connection to release
     */
    public void releaseConnection(Connection conn) {
        if (conn != null) {
            try {
                // Remove from active connections map
                activeConnections.remove(conn);
                
                // Reset auto-commit to default (true)
                if (!conn.getAutoCommit()) {
                    conn.setAutoCommit(true);
                }
                
                // Close the connection (returns it to the pool)
                conn.close();
                LOGGER.log(Level.FINE, "Released database connection, active connections: {0}", activeConnections.size());
            } catch (SQLException e) {
                LOGGER.log(Level.WARNING, "Error releasing database connection", e);
            }
        }
    }
    
    /**
     * Gets the list of securities available for lending.
     *
     * @return list of available securities
     * @throws SQLException if a database access error occurs
     */
    public List<Security> getAvailableSecurities() throws SQLException {
        List<Security> securities = new ArrayList<>();
        
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement()) {
            
            stmt.setQueryTimeout(QUERY_TIMEOUT);
            
            try (ResultSet rs = stmt.executeQuery(SQL_AVAILABLE_SECURITIES)) {
                while (rs.next()) {
                    Security security = new Security();
                    security.setSecurityId(rs.getLong("security_id"));
                    security.setIsin(rs.getString("isin"));
                    security.setTicker(rs.getString("ticker"));
                    security.setDescription(rs.getString("description"));
                    security.setCurrency(rs.getString("currency"));
                    security.setMarketId(rs.getInt("market_id"));
                    security.setInventoryId(rs.getLong("inventory_id"));
                    security.setQuantityAvailable(rs.getBigDecimal("quantity_available"));
                    security.setQuantityOnLoan(rs.getBigDecimal("quantity_on_loan"));
                    security.setLastPrice(rs.getBigDecimal("last_price"));
                    security.setMarketValue(rs.getBigDecimal("market_value"));
                    
                    securities.add(security);
                }
            }
        }
        
        LOGGER.log(Level.INFO, "Retrieved {0} available securities", securities.size());
        return securities;
    }
    
    /**
     * Retrieves counterparty limit information.
     *
     * @param counterpartyId the ID of the counterparty
     * @return counterparty limit information or null if not found
     * @throws SQLException if a database access error occurs
     */
    public CounterpartyLimits getCounterpartyLimits(long counterpartyId) throws SQLException {
        try (Connection conn = getConnection();
             PreparedStatement pstmt = conn.prepareStatement(SQL_COUNTERPARTY_LIMITS)) {
            
            pstmt.setLong(1, counterpartyId);
            pstmt.setQueryTimeout(QUERY_TIMEOUT);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    CounterpartyLimits limits = new CounterpartyLimits();
                    limits.setCounterpartyId(rs.getLong("counterparty_id"));
                    limits.setCreditRating(rs.getString("credit_rating"));
                    limits.setMaxExposure(rs.getBigDecimal("max_exposure"));
                    limits.setCurrentExposure(rs.getBigDecimal("current_exposure"));
                    limits.setMaxSingleLoan(rs.getBigDecimal("max_single_loan"));
                    limits.setBaseHaircut(rs.getBigDecimal("base_haircut"));
                    limits.setMarginRequirement(rs.getBigDecimal("margin_requirement"));
                    
                    LOGGER.log(Level.INFO, "Retrieved limits for counterparty ID: {0}", counterpartyId);
                    return limits;
                } else {
                    LOGGER.log(Level.WARNING, "No limits found for counterparty ID: {0}", counterpartyId);
                    return null;
                }
            }
        }
    }
    
    /**
     * Records a new stock loan transaction.
     *
     * @param loan the loan information to record
     * @return the generated loan ID, or -1 if the operation failed
     * @throws SQLException if a database access error occurs
     */
    public long recordLoan(StockLoan loan) throws SQLException {
        Connection conn = null;
        long loanId = -1;
        
        try {
            conn = getConnection();
            conn.setAutoCommit(false);  // Start transaction
            
            // Check inventory availability
            try (PreparedStatement checkStmt = conn.prepareStatement(
                    "SELECT quantity_available FROM securities_inventory WHERE security_id = ?")) {
                checkStmt.setLong(1, loan.getSecurityId());
                checkStmt.setQueryTimeout(QUERY_TIMEOUT);
                
                try (ResultSet rs = checkStmt.executeQuery()) {
                    if (!rs.next() || rs.getBigDecimal("quantity_available").compareTo(loan.getQuantity()) < 0) {
                        LOGGER.log(Level.WARNING, "Insufficient inventory for security ID: {0}", loan.getSecurityId());
                        conn.rollback();
                        return -1;
                    }
                }
            }
            
            // Update inventory
            try (PreparedStatement updateStmt = conn.prepareStatement(SQL_UPDATE_INVENTORY)) {
                updateStmt.setBigDecimal(1, loan.getQuantity());
                updateStmt.setBigDecimal(2, loan.getQuantity());
                updateStmt.setLong(3, loan.getSecurityId());
                updateStmt.setBigDecimal(4, loan.getQuantity());
                updateStmt.setQueryTimeout(QUERY_TIMEOUT);
                
                int rowsUpdated = updateStmt.executeUpdate();
                if (rowsUpdated == 0) {
                    LOGGER.log(Level.WARNING, "Failed to update inventory for security ID: {0}", loan.getSecurityId());
                    conn.rollback();
                    return -1;
                }
            }
            
            // Insert loan record
            try (PreparedStatement insertStmt = conn.prepareStatement(SQL_RECORD_LOAN, Statement.RETURN_GENERATED_KEYS)) {
                int index = 1;
                insertStmt.setString(index++, loan.getLoanRef());
                insertStmt.setLong(index++, loan.getCounterpartyId());
                insertStmt.setLong(index++, loan.getSecurityId());
                insertStmt.setBigDecimal(index++, loan.getQuantity());
                insertStmt.setString(index++, loan.getRateType());
                insertStmt.setBigDecimal(index++, loan.getLoanRate());
                insertStmt.setInt(index++, loan.getTermDays());
                insertStmt.setDate(index++, java.sql.Date.valueOf(loan.getStartDate()));
                insertStmt.setDate(index++, java.sql.Date.valueOf(loan.getEndDate()));
                insertStmt.setString(index++, loan.getBaseCurrency());
                insertStmt.setBigDecimal(index++, loan.getMarketValue());
                insertStmt.setLong(index++, loan.getCollateralId());
                insertStmt.setBigDecimal(index++, loan.getHaircut());
                insertStmt.setString(index++, loan.getSettlementStatus());
                insertStmt.setDate(index++, java.sql.Date.valueOf(LocalDate.now()));
                insertStmt.setTimestamp(index++, new java.sql.Timestamp(System.currentTimeMillis()));
                insertStmt.setString(index++, loan.getEnteredBy());
                insertStmt.setString(index++, loan.getSettlementLocation());
                insertStmt.setString(index++, loan.getFeeAccrualBasis());
                insertStmt.setQueryTimeout(QUERY_TIMEOUT);
                
                int rowsInserted = insertStmt.executeUpdate();
                
                if (rowsInserted > 0) {
                    try (ResultSet generatedKeys = insertStmt.getGeneratedKeys()) {
                        if (generatedKeys.next()) {
                            loanId = generatedKeys.getLong(1);
                        }
                    }
                }
            }
            
            // If successful, commit the transaction
            if (loanId > 0) {
                conn.commit();
                LOGGER.log(Level.INFO, "Loan recorded successfully, loan ID: {0}", loanId);
            } else {
                conn.rollback();
                LOGGER.log(Level.WARNING, "Failed to record loan");
            }
            
            return loanId;
        } catch (SQLException e) {
            LOGGER.log(Level.SEVERE, "Error recording loan", e);
            if (conn != null) {
                try {
                    conn.rollback();
                    LOGGER.log(Level.INFO, "Transaction rolled back");
                } catch (SQLException ex) {
                    LOGGER.log(Level.SEVERE, "Error during rollback", ex);
                }
            }
            throw e;
        } finally {
            if (conn != null) {
                try {
                    conn.setAutoCommit(true);  // Reset auto-commit
                    releaseConnection(conn);
                } catch (SQLException e) {
                    LOGGER.log(Level.WARNING, "Error resetting auto-commit", e);
                }
            }
        }
    }
    
    /**
     * Retrieves active stock loans.
     *
     * @return list of active loans
     * @throws SQLException if a database access error occurs
     */
    public List<StockLoan> getActiveLoans() throws SQLException {
        List<StockLoan> loans = new ArrayList<>();
        LocalDate today = LocalDate.now();
        
        try (Connection conn = getConnection();
             PreparedStatement pstmt = conn.prepareStatement(SQL_ACTIVE_LOANS)) {
            
            pstmt.setDate(1, java.sql.Date.valueOf(today));
            pstmt.setQueryTimeout(QUERY_TIMEOUT);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    StockLoan loan = new StockLoan();
                    loan.setLoanId(rs.getLong("loan_id"));
                    loan.setLoanRef(rs.getString("loan_ref"));
                    loan.setCounterpartyId(rs.getLong("counterparty_id"));
                    loan.setCounterpartyName(rs.getString("counterparty_name"));
                    loan.setSecurityId(rs.getLong("security_id"));
                    loan.setTicker(rs.getString("ticker"));
                    loan.setIsin(rs.getString("isin"));
                    loan.setQuantity(rs.getBigDecimal("quantity"));
                    loan.setLoanRate(rs.getBigDecimal("loan_rate"));
                    loan.setStartDate(rs.getDate("start_date").toLocalDate());
                    loan.setEndDate(rs.getDate("end_date").toLocalDate());
                    loan.setMarketValue(rs.getBigDecimal("market_value"));
                    loan.setSettlementStatus(rs.getString("settlement_status"));
                    
                    loans.add(loan);
                }
            }
        }
        
        LOGGER.log(Level.INFO, "Retrieved {0} active loans", loans.size());
        return loans;
    }
    
    /**
     * Recalls a stock loan.
     *
     * @param loanId the ID of the loan to recall
     * @param reason the reason for recall
     * @param initiatedBy the user initiating the recall
     * @return true if recall was successful, false otherwise
     * @throws SQLException if a database access error occurs
     */
    public boolean recallLoan(long loanId, String reason, String initiatedBy) throws SQLException {
        try (Connection conn = getConnection();
             PreparedStatement pstmt = conn.prepareStatement(SQL_RECALL_LOAN)) {
            
            LocalDate today = LocalDate.now();
            
            pstmt.setDate(1, java.sql.Date.valueOf(today));
            pstmt.setString(2, reason);
            pstmt.setString(3, initiatedBy);
            pstmt.setLong(4, loanId);
            pstmt.setQueryTimeout(QUERY_TIMEOUT);
            
            int rowsUpdated = pstmt.executeUpdate();
            boolean success = rowsUpdated > 0;
            
            if (success) {
                LOGGER.log(Level.INFO, "Loan ID {0} recalled successfully", loanId);
            } else {
                LOGGER.log(Level.WARNING, "Failed to recall loan ID {0}", loanId);
            }
            
            return success;
        }
    }
    
    /**
     * Calculates fees for a loan for a specific date range.
     *
     * @param loanId the loan ID
     * @param startDate the start date for fee calculation
     * @param endDate the end date for fee calculation
     * @return the calculated fee amount
     * @throws SQLException if a database access error occurs
     */
    public BigDecimal calculateLoanFees(long loanId, LocalDate startDate, LocalDate endDate) throws SQLException {
        try (Connection conn = getConnection();
             CallableStatement cstmt = conn.prepareCall(SQL_CALCULATE_FEES)) {
            
            cstmt.setLong(1, loanId);
            cstmt.setDate(2, java.sql.Date.valueOf(startDate));
            cstmt.setDate(3, java.sql.Date.valueOf(endDate));
            cstmt.registerOutParameter(4, java.sql.Types.DECIMAL);
            cstmt.setQueryTimeout(QUERY_TIMEOUT);
            
            cstmt.execute();
            
            BigDecimal feeAmount = cstmt.getBigDecimal(4);
            LOGGER.log(Level.INFO, "Calculated fee for loan ID {0}: {1}", new Object[]{loanId, feeAmount});
            
            return feeAmount;
        }
    }
    
    /**
     * Executes a batch update for end-of-day processing.
     *
     * @param processingDate the date for which to run processing
     * @return the number of records processed
     * @throws SQLException if a database access error occurs
     */
    public int runEndOfDayProcessing(LocalDate processingDate) throws SQLException {
        Connection conn = null;
        int recordsProcessed = 0;
        
        try {
            conn = getConnection();
            conn.setAutoCommit(false);  // Start transaction
            
            // Step 1: Update market values
            try (CallableStatement cstmt = conn.prepareCall("{call update_market_values(?)}")) {
                cstmt.setDate(1, java.sql.Date.valueOf(processingDate));
                cstmt.setQueryTimeout(QUERY_TIMEOUT * 2);  // Double timeout for batch operations
                
                boolean hasResults = cstmt.execute();
                if (hasResults) {
                    try (ResultSet rs = cstmt.getResultSet()) {
                        if (rs.next()) {
                            recordsProcessed += rs.getInt(1);
                        }
                    }
                }
            }
            
            // Step 2: Calculate fees
            try (CallableStatement cstmt = conn.prepareCall("{call calculate_daily_fees(?)}")) {
                cstmt.setDate(1, java.sql.Date.valueOf(processingDate));
                cstmt.setQueryTimeout(QUERY_TIMEOUT * 2);
                
                boolean hasResults = cstmt.execute();
                if (hasResults) {
                    try (ResultSet rs = cstmt.getResultSet()) {
                        if (rs.next()) {
                            recordsProcessed += rs.getInt(1);
                        }
                    }
                }
            }
            
            // Step 3: Update counterparty exposure
            try (CallableStatement cstmt = conn.prepareCall("{call update_counterparty_exposure(?)}")) {
                cstmt.setDate(1, java.sql.Date.valueOf(processingDate));
                cstmt.setQueryTimeout(QUERY_TIMEOUT * 2);
                
                boolean hasResults = cstmt.execute();
                if (hasResults) {
                    try (ResultSet rs = cstmt.getResultSet()) {
                        if (rs.next()) {
                            recordsProcessed += rs.getInt(1);
                        }
                    }
                }
            }
            
            // Step 4: Generate reports
            try (CallableStatement cstmt = conn.prepareCall("{call generate_daily_reports(?)}")) {
                cstmt.setDate(1, java.sql.Date.valueOf(processingDate));
                cstmt.setQueryTimeout(QUERY_TIMEOUT * 2);
                
                cstmt.execute();
            }
            
            // Commit the transaction
            conn.commit();
            LOGGER.log(Level.INFO, "End of day processing completed for {0}, records processed: {1}", 
                      new Object[]{processingDate, recordsProcessed});
            
            return recordsProcessed;
        } catch (SQLException e) {
            LOGGER.log(Level.SEVERE, "Error during end of day processing", e);
            if (conn != null) {
                try {
                    conn.rollback();
                    LOGGER.log(Level.INFO, "Transaction rolled back");
                } catch (SQLException ex) {
                    LOGGER.log(Level.SEVERE, "Error during rollback", ex);
                }
            }
            throw e;
        } finally {
            if (conn != null) {
                try {
                    conn.setAutoCommit(true);  // Reset auto-commit
                    releaseConnection(conn);
                } catch (SQLException e) {
                    LOGGER.log(Level.WARNING, "Error resetting auto-commit", e);
                }
            }
        }
    }
    
    /**
     * Searches for securities by various criteria.
     *
     * @param ticker ticker symbol (can be partial)
     * @param isin ISIN code (can be partial)
     * @param marketId market ID (optional, use 0 to ignore)
     * @return list of matching securities
     * @throws SQLException if a database access error occurs
     */
    public List<Security> searchSecurities(String ticker, String isin, int marketId) throws SQLException {
        List<Security> results = new ArrayList<>();
        StringBuilder sql = new StringBuilder(
            "SELECT s.security_id, s.isin, s.ticker, s.description, s.currency, " +
            "s.market_id, i.inventory_id, i.quantity_available, i.quantity_on_loan, " +
            "i.last_price, i.market_value " +
            "FROM securities s " +
            "INNER JOIN securities_inventory i ON s.security_id = i.security_id " +
            "WHERE 1=1"
        );
        
        List<Object> params = new ArrayList<>();
        
        if (ticker != null && !ticker.trim().isEmpty()) {
            sql.append(" AND s.ticker LIKE ?");
            params.add("%" + ticker.trim() + "%");
        }
        
        if (isin != null && !isin.trim().isEmpty()) {
            sql.append(" AND s.isin LIKE ?");
            params.add("%" + isin.trim() + "%");
        }
        
        if (marketId > 0) {
            sql.append(" AND s.market_id = ?");
            params.add(marketId);
        }
        
        sql.append(" ORDER BY s.market_id, s.ticker");
        
        try (Connection conn = getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql.toString())) {
            
            // Set parameters
            for (int i = 0; i < params.size(); i++) {
                pstmt.setObject(i + 1, params.get(i));
            }
            
            pstmt.setQueryTimeout(QUERY_TIMEOUT);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    Security security = new Security();
                    security.setSecurityId(rs.getLong("security_id"));
                    security.setIsin(rs.getString("isin"));
                    security.setTicker(rs.getString("ticker"));
                    security.setDescription(rs.getString("description"));
                    security.setCurrency(rs.getString("currency"));
                    security.setMarketId(rs.getInt("market_id"));
                    security.setInventoryId(rs.getLong("inventory_id"));
                    security.setQuantityAvailable(rs.getBigDecimal("quantity_available"));
                    security.setQuantityOnLoan(rs.getBigDecimal("quantity_on_loan"));
                    security.setLastPrice(rs.getBigDecimal("last_price"));
                    security.setMarketValue(rs.getBigDecimal("market_value"));
                    
                    results.add(security);
                }
            }
        }
        
        LOGGER.log(Level.INFO, "Search found {0} securities", results.size());
        return results;
    }
    
    /**
     * Tests the database connection.
     *
     * @return true if connection is successful, false otherwise
     */
    public boolean testConnection() {
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement()) {
            
            stmt.setQueryTimeout(QUERY_TIMEOUT);
            stmt.execute("SELECT 1");
            
            LOGGER.log(Level.INFO, "Database connection test successful");
            return true;
        } catch (SQLException e) {
            LOGGER.log(Level.SEVERE, "Database connection test failed", e);
            return false;
        }
    }
    
    /**
     * Retrieves the total market value of all securities on loan.
     *
     * @return total market value of securities on loan
     * @throws SQLException if a database access error occurs
     */
    public BigDecimal getTotalMarketValueOnLoan() throws SQLException {
        String sql = "SELECT SUM(market_value) AS total_market_value FROM stock_loans WHERE settlement_status = 'SETTLED'";
        
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            
            if (rs.next()) {
                BigDecimal totalMarketValue = rs.getBigDecimal("total_market_value");
                LOGGER.log(Level.INFO, "Total market value on loan: {0}", totalMarketValue);
                return totalMarketValue;
            } else {
                return BigDecimal.ZERO;
            }
        }
    }

    /**
     * Retrieves the number of active loans for a specific counterparty.
     *
     * @param counterpartyId the ID of the counterparty
     * @return the number of active loans
     * @throws SQLException if a database access error occurs
     */
    public int getActiveLoansCountForCounterparty(long counterpartyId) throws SQLException {
        String sql = "SELECT COUNT(*) AS active_loans_count FROM stock_loans WHERE counterparty_id = ? AND settlement_status = 'SETTLED'";
        
        try (Connection conn = getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            pstmt.setLong(1, counterpartyId);
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    int activeLoansCount = rs.getInt("active_loans_count");
                    LOGGER.log(Level.INFO, "Active loans count for counterparty ID {0}: {1}", new Object[]{counterpartyId, activeLoansCount});
                    return activeLoansCount;
                } else {
                    return 0;
                }
            }
        }
    }

    /**
     * Retrieves the average loan rate for all settled loans.
     *
     * @return the average loan rate
     * @throws SQLException if a database access error occurs
     */
    public BigDecimal getAverageLoanRate() throws SQLException {
        String sql = "SELECT AVG(loan_rate) AS average_loan_rate FROM stock_loans WHERE settlement_status = 'SETTLED'";
        
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            
            if (rs.next()) {
                BigDecimal averageLoanRate = rs.getBigDecimal("average_loan_rate");
                LOGGER.log(Level.INFO, "Average loan rate: {0}", averageLoanRate);
                return averageLoanRate;
            } else {
                return BigDecimal.ZERO;
            }
        }
    }

    /**
     * Retrieves the total quantity of securities available for lending.
     *
     * @return total quantity of securities available
     * @throws SQLException if a database access error occurs
     */
    public BigDecimal getTotalQuantityAvailable() throws SQLException {
        String sql = "SELECT SUM(quantity_available) AS total_quantity_available FROM securities_inventory";
        
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            
            if (rs.next()) {
                BigDecimal totalQuantityAvailable = rs.getBigDecimal("total_quantity_available");
                LOGGER.log(Level.INFO, "Total quantity available: {0}", totalQuantityAvailable);
                return totalQuantityAvailable;
            } else {
                return BigDecimal.ZERO;
            }
        }
    }

    /**
     * Retrieves the total number of counterparties.
     *
     * @return total number of counterparties
     * @throws SQLException if a database access error occurs
     */
    public int getTotalCounterparties() throws SQLException {
        String sql = "SELECT COUNT(*) AS total_counterparties FROM counterparties";
        
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            
            if (rs.next()) {
                int totalCounterparties = rs.getInt("total_counterparties");
                LOGGER.log(Level.INFO, "Total counterparties: {0}", totalCounterparties);
                return totalCounterparties;
            } else {
                return 0;
            }
        }
    }

    /**
     * Closes the GlobalStockLendingSystem and releases all resources.
     */
    @Override
    public void close() {
        for (Connection conn : activeConnections.keySet()) {
            releaseConnection(conn);
        }
        
        LOGGER.log(Level.INFO, "GlobalStockLendingSystem closed, all connections released");
    }
}