/**
 * @file SybaseDatabaseManager.cpp
 * @brief Implements secure CRUD operations for Sybase database access
 *
 * This module provides a secure and reliable interface for accessing 
 * Sybase databases within the enterprise financial environment.
 *
 * @author Financial Systems Team
 * @version 2.3.1
 * @date 2005-06-14
 *
 * Copyright (c) 2005 Global Financial Services, Inc.
 * All rights reserved.
 */

#include "SybaseDatabaseManager.h"
#include "Logger.h"
#include "ConfigurationManager.h"
#include "SecurityManager.h"
#include "ErrorCodes.h"

// Namespace to avoid global pollution
namespace FinancialSystems {
namespace Database {

/**
 * @brief Constructor initializes database connection parameters
 *
 * @param configFile Path to the configuration file
 * @throws DatabaseException if initialization fails
 */
SybaseDatabaseManager::SybaseDatabaseManager(const std::string& configFile)
    : m_context(NULL),
      m_connection(NULL),
      m_command(NULL),
      m_isConnected(false),
      m_transactionActive(false)
{
    TRACE_ENTER("SybaseDatabaseManager::SybaseDatabaseManager");
    
    try {
        // Load configuration
        m_config = ConfigurationManager::getInstance().loadDatabaseConfig(configFile);
        
        // Initialize the CS context
        CS_RETCODE rc = cs_ctx_alloc(CS_VERSION_125, &m_context);
        if (rc != CS_SUCCEED) {
            TRACE_ERROR("Failed to allocate CS context");
            throw DatabaseException(ERR_DB_CONTEXT_INIT, "Failed to allocate CS context");
        }
        
        // Initialize CT library
        rc = ct_init(m_context, CS_VERSION_125);
        if (rc != CS_SUCCEED) {
            TRACE_ERROR("Failed to initialize client library");
            cs_ctx_drop(m_context);
            m_context = NULL;
            throw DatabaseException(ERR_DB_LIB_INIT, "Failed to initialize client library");
        }
        
        // Set up error handlers
        setupErrorHandlers();
    }
    catch (ConfigException& ex) {
        TRACE_ERROR("Configuration error: " << ex.what());
        cleanup();
        throw DatabaseException(ERR_DB_CONFIG, "Configuration error: " + std::string(ex.what()));
    }
    catch (DatabaseException&) {
        // Just rethrow
        throw;
    }
    catch (std::exception& ex) {
        TRACE_ERROR("Unexpected error during initialization: " << ex.what());
        cleanup();
        throw DatabaseException(ERR_DB_INIT, "Unexpected error: " + std::string(ex.what()));
    }
    
    TRACE_EXIT("SybaseDatabaseManager::SybaseDatabaseManager");
}

/**
 * @brief Destructor ensures clean disconnection and resource release
 */
SybaseDatabaseManager::~SybaseDatabaseManager() {
    TRACE_ENTER("SybaseDatabaseManager::~SybaseDatabaseManager");
    
    try {
        if (m_isConnected) {
            disconnect();
        }
    }
    catch (DatabaseException& ex) {
        TRACE_ERROR("Error during disconnect: " << ex.what());
    }
    
    cleanup();
    
    TRACE_EXIT("SybaseDatabaseManager::~SybaseDatabaseManager");
}

/**
 * @brief Establishes a connection to the Sybase database
 *
 * @returns true if connection successful, false otherwise
 * @throws DatabaseException for serious errors
 */
bool SybaseDatabaseManager::connect() {
    TRACE_ENTER("SybaseDatabaseManager::connect");
    
    if (m_isConnected) {
        TRACE_INFO("Already connected to database");
        TRACE_EXIT("SybaseDatabaseManager::connect - Already connected");
        return true;
    }
    
    CS_RETCODE rc;
    
    try {
        // Allocate connection structure
        rc = ct_con_alloc(m_context, &m_connection);
        if (rc != CS_SUCCEED) {
            TRACE_ERROR("Failed to allocate connection structure");
            throw DatabaseException(ERR_DB_CONNECT, "Failed to allocate connection structure");
        }
        
        // Get secure credentials from security manager
        std::string username, password;
        SecurityManager::getInstance().getCredentials(
            m_config.credentialId, username, password);
        
        // Set login credentials
        rc = ct_con_props(m_connection, CS_SET, CS_USERNAME, 
                         (CS_VOID*)username.c_str(), CS_NULLTERM, NULL);
        if (rc != CS_SUCCEED) {
            TRACE_ERROR("Failed to set username");
            ct_con_drop(m_connection);
            m_connection = NULL;
            throw DatabaseException(ERR_DB_CONNECT, "Failed to set username");
        }
        
        rc = ct_con_props(m_connection, CS_SET, CS_PASSWORD, 
                         (CS_VOID*)password.c_str(), CS_NULLTERM, NULL);
        if (rc != CS_SUCCEED) {
            TRACE_ERROR("Failed to set password");
            ct_con_drop(m_connection);
            m_connection = NULL;
            throw DatabaseException(ERR_DB_CONNECT, "Failed to set password");
        }
        
        // Connect to the server
        rc = ct_connect(m_connection, m_config.server.c_str(), CS_NULLTERM);
        if (rc != CS_SUCCEED) {
            TRACE_ERROR("Failed to connect to server: " << m_config.server);
            ct_con_drop(m_connection);
            m_connection = NULL;
            throw DatabaseException(ERR_DB_CONNECT, 
                                   "Failed to connect to server: " + m_config.server);
        }
        
        // Switch to the specified database
        std::string useDbCmd = "USE " + m_config.database;
        executeCommand(useDbCmd);
        
        m_isConnected = true;
        TRACE_INFO("Successfully connected to database " << m_config.database);
    }
    catch (SecurityException& ex) {
        TRACE_ERROR("Security error during connect: " << ex.what());
        if (m_connection) {
            ct_con_drop(m_connection);
            m_connection = NULL;
        }
        throw DatabaseException(ERR_DB_SECURITY, "Security error: " + std::string(ex.what()));
    }
    catch (DatabaseException&) {
        // Clean up and rethrow
        if (m_connection) {
            ct_con_drop(m_connection);
            m_connection = NULL;
        }
        throw;
    }
    
    TRACE_EXIT("SybaseDatabaseManager::connect");
    return true;
}

/**
 * @brief Disconnects from the Sybase database
 * 
 * @returns true if disconnection successful, false otherwise
 * @throws DatabaseException for serious errors
 */
bool SybaseDatabaseManager::disconnect() {
    TRACE_ENTER("SybaseDatabaseManager::disconnect");
    
    if (!m_isConnected) {
        TRACE_INFO("Not connected to database");
        TRACE_EXIT("SybaseDatabaseManager::disconnect - Not connected");
        return true;
    }
    
    try {
        // Rollback any active transaction
        if (m_transactionActive) {
            rollbackTransaction();
        }
        
        // Close the connection
        CS_RETCODE rc = ct_close(m_connection, CS_UNUSED);
        if (rc != CS_SUCCEED) {
            TRACE_ERROR("Failed to close connection");
            throw DatabaseException(ERR_DB_DISCONNECT, "Failed to close connection");
        }
        
        // Drop the connection
        rc = ct_con_drop(m_connection);
        if (rc != CS_SUCCEED) {
            TRACE_ERROR("Failed to drop connection");
            throw DatabaseException(ERR_DB_DISCONNECT, "Failed to drop connection");
        }
        
        m_connection = NULL;
        m_isConnected = false;
        
        TRACE_INFO("Successfully disconnected from database");
    }
    catch (DatabaseException&) {
        // Just rethrow
        throw;
    }
    
    TRACE_EXIT("SybaseDatabaseManager::disconnect");
    return true;
}

/**
 * @brief Creates a new customer record in the database
 *
 * @param customer Customer object containing data to insert
 * @returns true if insertion successful, false otherwise
 * @throws DatabaseException for serious errors
 */
bool SybaseDatabaseManager::createCustomer(const Customer& customer) {
    TRACE_ENTER("SybaseDatabaseManager::createCustomer");
    
    if (!m_isConnected) {
        TRACE_ERROR("Not connected to database");
        throw DatabaseException(ERR_DB_NOT_CONNECTED, "Not connected to database");
    }
    
    try {
        // Validate customer data
        if (!validateCustomerData(customer)) {
            TRACE_ERROR("Invalid customer data");
            return false;
        }
        
        // Prepare parameterized query to prevent SQL injection
        std::string sql = "INSERT INTO Customers (ID, Name, Address, Balance) VALUES (?, ?, ?, ?)";
        
        // Allocate a command structure
        if (ct_cmd_alloc(m_connection, &m_command) != CS_SUCCEED) {
            TRACE_ERROR("Failed to allocate command structure");
            throw DatabaseException(ERR_DB_COMMAND, "Failed to allocate command structure");
        }
        
        // Initialize the command
        if (ct_command(m_command, CS_LANG_CMD, const_cast<char*>(sql.c_str()), 
                     CS_NULLTERM, CS_UNUSED) != CS_SUCCEED) {
            TRACE_ERROR("Failed to initialize command");
            ct_cmd_drop(m_command);
            m_command = NULL;
            throw DatabaseException(ERR_DB_COMMAND, "Failed to initialize command");
        }
        
        // Set up parameter data
        CS_DATAFMT paramFmt;
        CS_INT customerId = customer.getId();
        CS_CHAR name[MAX_NAME_LENGTH + 1];
        CS_CHAR address[MAX_ADDRESS_LENGTH + 1];
        CS_FLOAT balance = static_cast<CS_FLOAT>(customer.getBalance());
        
        // Copy strings safely to prevent buffer overflows
        strncpy(reinterpret_cast<char*>(name), customer.getName().c_str(), MAX_NAME_LENGTH);
        name[MAX_NAME_LENGTH] = '\0';
        
        strncpy(reinterpret_cast<char*>(address), customer.getAddress().c_str(), MAX_ADDRESS_LENGTH);
        address[MAX_ADDRESS_LENGTH] = '\0';
        
        // Bind parameters
        bindIntParameter(customerId);
        bindStringParameter(name, strlen(reinterpret_cast<char*>(name)));
        bindStringParameter(address, strlen(reinterpret_cast<char*>(address)));
        bindFloatParameter(balance);
        
        // Execute the command
        CS_INT rowsAffected = executeCommandWithRowCount();
        
        TRACE_INFO("Customer created with ID " << customer.getId() << 
                  ", affected " << rowsAffected << " row(s)");
        
        // Return success if at least one row was affected
        return (rowsAffected > 0);
    }
    catch (DatabaseException&) {
        // Clean up if necessary
        cleanupCommand();
        throw;
    }
    catch (std::exception& ex) {
        // Clean up if necessary
        cleanupCommand();
        TRACE_ERROR("Unexpected error creating customer: " << ex.what());
        throw DatabaseException(ERR_DB_COMMAND, "Unexpected error: " + std::string(ex.what()));
    }
    
    TRACE_EXIT("SybaseDatabaseManager::createCustomer");
}

/**
 * @brief Reads customer data from the database by ID
 *
 * @param id The customer ID to look up
 * @param customer Output parameter for the customer data
 * @returns true if customer found, false if not found
 * @throws DatabaseException for serious errors
 */
bool SybaseDatabaseManager::readCustomer(int id, Customer& customer) {
    TRACE_ENTER("SybaseDatabaseManager::readCustomer");
    
    if (!m_isConnected) {
        TRACE_ERROR("Not connected to database");
        throw DatabaseException(ERR_DB_NOT_CONNECTED, "Not connected to database");
    }
    
    try {
        // Prepare the SQL statement with parameter to prevent SQL injection
        std::string sql = "SELECT ID, Name, Address, Balance FROM Customers WHERE ID = ?";
        
        // Allocate and prepare command
        if (!prepareCommand(sql)) {
            throw DatabaseException(ERR_DB_COMMAND, "Failed to prepare command");
        }
        
        // Bind the ID parameter
        CS_INT customerId = id;
        bindIntParameter(customerId);
        
        // Execute the command
        if (ct_send(m_command) != CS_SUCCEED) {
            TRACE_ERROR("Failed to send command");
            cleanupCommand();
            throw DatabaseException(ERR_DB_COMMAND, "Failed to send command");
        }
        
        // Process the results
        CS_INT resultType;
        CS_RETCODE rcResults;
        bool customerFound = false;
        
        while ((rcResults = ct_results(m_command, &resultType)) == CS_SUCCEED) {
            if (resultType == CS_ROW_RESULT) {
                // Set up result bindings
                CS_DATAFMT resFmt;
                CS_INT resId;
                CS_CHAR resName[MAX_NAME_LENGTH + 1];
                CS_CHAR resAddress[MAX_ADDRESS_LENGTH + 1];
                CS_FLOAT resBalance;
                
                // Bind result columns
                setupResultBinding(1, CS_INT_TYPE, &resId, sizeof(resId));
                setupResultBinding(2, CS_CHAR_TYPE, resName, MAX_NAME_LENGTH);
                setupResultBinding(3, CS_CHAR_TYPE, resAddress, MAX_ADDRESS_LENGTH);
                setupResultBinding(4, CS_FLOAT_TYPE, &resBalance, sizeof(resBalance));
                
                // Fetch the data
                if (ct_fetch(m_command, CS_UNUSED, CS_UNUSED, CS_UNUSED) == CS_SUCCEED) {
                    // Fill the customer object with fetched data
                    customer.setId(resId);
                    customer.setName(reinterpret_cast<char*>(resName));
                    customer.setAddress(reinterpret_cast<char*>(resAddress));
                    customer.setBalance(static_cast<double>(resBalance));
                    
                    customerFound = true;
                    TRACE_INFO("Customer found with ID " << resId);
                    break;
                }
            }
        }
        
        // Process remaining results to complete the command
        processRemainingResults();
        
        // Clean up
        cleanupCommand();
        
        TRACE_EXIT("SybaseDatabaseManager::readCustomer");
        return customerFound;
    }
    catch (DatabaseException&) {
        cleanupCommand();
        throw;
    }
    catch (std::exception& ex) {
        cleanupCommand();
        TRACE_ERROR("Unexpected error reading customer: " << ex.what());
        throw DatabaseException(ERR_DB_COMMAND, "Unexpected error: " + std::string(ex.what()));
    }
}

/**
 * @brief Updates customer balance in the database
 *
 * @param id Customer ID to update
 * @param newBalance New balance value
 * @returns true if update successful, false if customer not found
 * @throws DatabaseException for serious errors
 */
bool SybaseDatabaseManager::updateCustomerBalance(int id, double newBalance) {
    TRACE_ENTER("SybaseDatabaseManager::updateCustomerBalance");
    
    if (!m_isConnected) {
        TRACE_ERROR("Not connected to database");
        throw DatabaseException(ERR_DB_NOT_CONNECTED, "Not connected to database");
    }
    
    try {
        // Prepare the SQL statement with parameters to prevent SQL injection
        std::string sql = "UPDATE Customers SET Balance = ? WHERE ID = ?";
        
        // Allocate and prepare command
        if (!prepareCommand(sql)) {
            throw DatabaseException(ERR_DB_COMMAND, "Failed to prepare command");
        }
        
        // Bind parameters
        CS_FLOAT balance = static_cast<CS_FLOAT>(newBalance);
        CS_INT customerId = id;
        
        bindFloatParameter(balance);
        bindIntParameter(customerId);
        
        // Execute command and get rows affected
        CS_INT rowsAffected = executeCommandWithRowCount();
        
        TRACE_INFO("Customer balance updated for ID " << id << 
                  ", affected " << rowsAffected << " row(s)");
        
        // Return success if at least one row was affected
        return (rowsAffected > 0);
    }
    catch (DatabaseException&) {
        cleanupCommand();
        throw;
    }
    catch (std::exception& ex) {
        cleanupCommand();
        TRACE_ERROR("Unexpected error updating customer: " << ex.what());
        throw DatabaseException(ERR_DB_COMMAND, "Unexpected error: " + std::string(ex.what()));
    }
    
    TRACE_EXIT("SybaseDatabaseManager::updateCustomerBalance");
}

/**
 * @brief Deletes a customer record from the database
 *
 * @param id Customer ID to delete
 * @returns true if deletion successful, false if customer not found
 * @throws DatabaseException for serious errors
 */
bool SybaseDatabaseManager::deleteCustomer(int id) {
    TRACE_ENTER("SybaseDatabaseManager::deleteCustomer");
    
    if (!m_isConnected) {
        TRACE_ERROR("Not connected to database");
        throw DatabaseException(ERR_DB_NOT_CONNECTED, "Not connected to database");
    }
    
    try {
        // Use parameterized query for safety
        std::string sql = "DELETE FROM Customers WHERE ID = ?";
        
        // Allocate and prepare command
        if (!prepareCommand(sql)) {
            throw DatabaseException(ERR_DB_COMMAND, "Failed to prepare command");
        }
        
        // Bind parameter
        CS_INT customerId = id;
        bindIntParameter(customerId);
        
        // Execute command and get rows affected
        CS_INT rowsAffected = executeCommandWithRowCount();
        
        TRACE_INFO("Customer deleted with ID " << id << 
                  ", affected " << rowsAffected << " row(s)");
        
        // Return success if at least one row was affected
        return (rowsAffected > 0);
    }
    catch (DatabaseException&) {
        cleanupCommand();
        throw;
    }
    catch (std::exception& ex) {
        cleanupCommand();
        TRACE_ERROR("Unexpected error deleting customer: " << ex.what());
        throw DatabaseException(ERR_DB_COMMAND, "Unexpected error: " + std::string(ex.what()));
    }
    
    TRACE_EXIT("SybaseDatabaseManager::deleteCustomer");
}

/**
 * @brief Sets up error handlers for Sybase client and server messages
 *
 * @private
 */
void SybaseDatabaseManager::setupErrorHandlers() {
    TRACE_ENTER("SybaseDatabaseManager::setupErrorHandlers");
    
    // Set up server message callback
    CS_RETCODE rc = ct_callback(m_context, NULL, CS_SET, CS_SERVERMSG_CB, 
                               (CS_VOID*)SybaseDatabaseManager::serverMessageHandler);
    if (rc != CS_SUCCEED) {
        TRACE_ERROR("Failed to install server message handler");
        throw DatabaseException(ERR_DB_CONFIG, "Failed to install server message handler");
    }
    
    // Set up client message callback
    rc = ct_callback(m_context, NULL, CS_SET, CS_CLIENTMSG_CB, 
                    (CS_VOID*)SybaseDatabaseManager::clientMessageHandler);
    if (rc != CS_SUCCEED) {
        TRACE_ERROR("Failed to install client message handler");
        throw DatabaseException(ERR_DB_CONFIG, "Failed to install client message handler");
    }
    
    TRACE_EXIT("SybaseDatabaseManager::setupErrorHandlers");
}

/**
 * @brief Handler for Sybase server error messages
 *
 * @param context Database context
 * @param connection Database connection
 * @param message Server message information
 * @return CS_SUCCEED always
 * @static
 */
CS_RETCODE SybaseDatabaseManager::serverMessageHandler(CS_CONTEXT* context, 
                                                      CS_CONNECTION* connection,
                                                      CS_SERVERMSG* message) {
    // Log the server error message
    if (message->severity > 10) {
        Logger::getInstance().logError(
            "Sybase Server Error: Msg %d, Level %d, State %d, Line %d: %s",
            message->msgnumber, message->severity, message->state,
            message->line, message->text);
    }
    else {
        Logger::getInstance().logWarning(
            "Sybase Server Message: Msg %d, Level %d, State %d, Line %d: %s",
            message->msgnumber, message->severity, message->state,
            message->line, message->text);
    }
    
    return CS_SUCCEED;
}

/**
 * @brief Handler for Sybase client library error messages
 *
 * @param context Database context
 * @param connection Database connection
 * @param message Client message information
 * @return CS_SUCCEED always
 * @static
 */
CS_RETCODE SybaseDatabaseManager::clientMessageHandler(CS_CONTEXT* context, 
                                                      CS_CONNECTION* connection,
                                                      CS_CLIENTMSG* message) {
    // Log the client error message
    Logger::getInstance().logError(
        "Sybase Client Error: Severity %d, Layer %d, Origin %d: %s",
        CS_SEVERITY(message->severity), CS_LAYER(message->msgnumber),
        CS_ORIGIN(message->msgnumber), message->msgstring);
    
    return CS_SUCCEED;
}

/**
 * @brief Executes a SQL command and returns affected row count
 *
 * @returns Number of rows affected by the command
 * @throws DatabaseException if command execution fails
 * @private
 */
CS_INT SybaseDatabaseManager::executeCommandWithRowCount() {
    TRACE_ENTER("SybaseDatabaseManager::executeCommandWithRowCount");
    
    if (ct_send(m_command) != CS_SUCCEED) {
        TRACE_ERROR("Failed to send command");
        throw DatabaseException(ERR_DB_COMMAND, "Failed to send command");
    }
    
    // Process the results
    CS_INT rowsAffected = 0;
    CS_INT resultType;
    CS_RETCODE rcResults;
    
    while ((rcResults = ct_results(m_command, &resultType)) == CS_SUCCEED) {
        if (resultType == CS_CMD_SUCCEED) {
            if (ct_res_info(m_command, CS_ROW_COUNT, &rowsAffected, 
                           CS_UNUSED, NULL) != CS_SUCCEED) {
                TRACE_WARNING("Failed to get rows affected");
            }
        }
        else if (resultType == CS_CMD_FAIL) {
            TRACE_ERROR("Command failed");
            throw DatabaseException(ERR_DB_COMMAND, "Command failed");
        }
    }
    
    // Check for error in results processing
    if (rcResults != CS_END_RESULTS) {
        TRACE_ERROR("Error processing results");
        throw DatabaseException(ERR_DB_RESULTS, "Error processing results");
    }
    
    TRACE_EXIT("SybaseDatabaseManager::executeCommandWithRowCount");
    return rowsAffected;
}

/**
 * @brief Cleans up resources when closing or destroying connection
 *
 * @private
 */
void SybaseDatabaseManager::cleanup() {
    TRACE_ENTER("SybaseDatabaseManager::cleanup");
    
    // Clean up command if necessary
    cleanupCommand();
    
    // Clean up connection if necessary
    if (m_connection) {
        ct_con_drop(m_connection);
        m_connection = NULL;
    }
    
    // Clean up context if necessary
    if (m_context) {
        ct_exit(m_context, CS_UNUSED);
        cs_ctx_drop(m_context);
        m_context = NULL;
    }
    
    TRACE_EXIT("SybaseDatabaseManager::cleanup");
}

/**
 * @brief Validates customer data before database operations
 *
 * @param customer The customer data to validate
 * @returns true if data is valid, false otherwise
 * @private
 */
bool SybaseDatabaseManager::validateCustomerData(const Customer& customer) {
    TRACE_ENTER("SybaseDatabaseManager::validateCustomerData");
    
    bool isValid = true;
    
    // Check ID
    if (customer.getId() <= 0) {
        TRACE_ERROR("Invalid customer ID: " << customer.getId());
        isValid = false;
    }
    
    // Check name
    if (customer.getName().empty() || customer.getName().length() > MAX_NAME_LENGTH) {
        TRACE_ERROR("Invalid customer name: " << customer.getName());
        isValid = false;
    }
    
    // Check address
    if (customer.getAddress().empty() || customer.getAddress().length() > MAX_ADDRESS_LENGTH) {
        TRACE_ERROR("Invalid customer address: " << customer.getAddress());
        isValid = false;
    }
    
    // Additional business rules can be added here
    
    TRACE_EXIT("SybaseDatabaseManager::validateCustomerData");
    return isValid;
}

} // namespace Database
} // namespace FinancialSystems