/**
 * @file complexfile.cpp
 * @brief Implements secure CRUD operations for Oracle database access
 * 
 * Converted from Sybase code by Sybase to Modern Database Converter
 * Conversion date: 2025-03-06 22:28:14
 */

/**
 * @file SybaseCrud.cpp
 * @brief Pointer-heavy Sybase operations with template metaprogramming
 */

#include "SybaseCrud.h"
#include <memory>
#include <functional>
#include <string.h>
#include <typeinfo>
#include <type_traits>

// Advanced template for type-safe parameter binding
template<typename T>
struct SybaseTypeTraits {
    // Default implementation will cause compile error
    static_assert(sizeof(T) == 0, "Unsupported type for Sybase binding");
};

// Specialization for int
template<>
struct SybaseTypeTraits<int> {
    static constexpr int datatype = CS_INT_TYPE;
    static constexpr const char* type_name = "CS_INT_TYPE";
    static constexpr size_t buffer_size = sizeof(int);
};

// Specialization for float
template<>
struct SybaseTypeTraits<float> {
    static constexpr int datatype = CS_FLOAT_TYPE;
    static constexpr const char* type_name = "CS_FLOAT_TYPE";
    static constexpr size_t buffer_size = sizeof(float);
};

// Specialization for double
template<>
struct SybaseTypeTraits<double> {
    static constexpr int datatype = CS_FLOAT_TYPE;
    static constexpr const char* type_name = "CS_FLOAT_TYPE";
    static constexpr size_t buffer_size = sizeof(float);
};

// Specialization for string (char*)
template<>
struct SybaseTypeTraits<char*> {
    static constexpr int datatype = CS_CHAR_TYPE;
    static constexpr const char* type_name = "CS_CHAR_TYPE";
    static constexpr size_t buffer_size = 0; // Dynamic size based on string length
};

// Function pointer type for error handlers
typedef Status (*CS_MSGFUNC_PTR)(Environment**, Connection**, void*);

// Smart pointer custom deleter for Sybase resources
struct SybaseDeleter {
    void operator()(Environment** ctx) const {
        if (ctx) {
            ct_exit(ctx, 0);
            /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
        }
    }
    
    void operator()(Connection** conn) const {
        if (conn) {
            ct_close(conn, 0);
            /* CONVERTED FROM SYBASE: ct_con_drop(conn) */
Environment::terminateConnection(conn)
/* END CONVERSION */;
        }
    }
    
    void operator()(Statement** cmd) const {
        if (cmd) {
            ct_cmd_drop(cmd);
        }
    }
};

// Custom unique_ptr types for Sybase resources
using ContextPtr = std::unique_ptr<Environment*, SybaseDeleter>;
using ConnectionPtr = std::unique_ptr<Connection*, SybaseDeleter>;
using CommandPtr = std::unique_ptr<Statement*, SybaseDeleter>;

// Main class for Sybase operations - nerdy implementation with pointers and templates
class SybaseManager {
private:
    // Smart pointers for automatic resource management
    ContextPtr m_pContext;
    ConnectionPtr m_pConnection;
    CommandPtr m_pCommand;
    
    // Bit flags for state management (using bit operations)
    uint16_t m_flags;
    
    // Bit positions in m_flags
    enum FlagBits {
        CONNECTED         = 0x0001,
        TRANSACTION_ACTIVE = 0x0002,
        COMMAND_ACTIVE    = 0x0004,
        DEBUG_MODE        = 0x0008,
        TRACE_ENABLED     = 0x0010
    };
    
    // Connection parameters
    char* m_pServer;
    char* m_pUsername;
    char* m_pPassword;
    char* m_pDatabase;
    
    // Function pointer to custom error callback
    std::function<void(const char*, int)> m_pfnErrorCallback;
    
    // Initialization list (used by constructor)
    struct init_params_t {
        const char* server;
        const char* username;
        const char* password;
        const char* database;
        uint16_t flags;
    };

public:
    // Constructor using pointer manipulation and custom initializer
    SybaseManager(const init_params_t& params)
        : m_pContext(nullptr),
          m_pConnection(nullptr),
          m_pCommand(nullptr),
          m_flags(params.flags)
    {
        // Allocate and copy strings using raw pointers
        size_t server_len = strlen(params.server) + 1;
        size_t username_len = strlen(params.username) + 1;
        size_t password_len = strlen(params.password) + 1;
        size_t database_len = strlen(params.database) + 1;
        
        m_pServer = new char[server_len];
        m_pUsername = new char[username_len];
        m_pPassword = new char[password_len];
        m_pDatabase = new char[database_len];
        
        // Use memcpy for raw memory copy
        memcpy(m_pServer, params.server, server_len);
        memcpy(m_pUsername, params.username, username_len);
        memcpy(m_pPassword, params.password, password_len);
        memcpy(m_pDatabase, params.database, database_len);
    }
    
    // Destructor with manual cleanup
    ~SybaseManager() {
        // Free all resources
        if (m_flags & CONNECTED) {
            disconnect();
        }
        
        // Free memory for connection strings
        delete[] m_pServer;
        delete[] m_pUsername;
        delete[] m_pPassword;
        delete[] m_pDatabase;
    }
    
    // Factory method returning raw pointer (old-school style)
    static SybaseManager* create(const char* server, const char* username, 
                                 const char* password, const char* database) {
        init_params_t params = {
            server, username, password, database, 0
        };
        
        return new SybaseManager(params);
    }
    
    // Connect method with raw pointers and error handling
    bool connect() {
        Environment** ctx = nullptr;
        Connection** conn = nullptr;
        Statement** cmd = nullptr;
        Status rc;
        
        // Check if already connected using bit flags
        if (m_flags & CONNECTED) {
            logError("Already connected", 0);
            return true;
        }
        
        try {
            // Initialize context
            rc = /* CONVERTED FROM SYBASE: cs_ctx_alloc(CS_VERSION_125, &ctx) */
ctx = Environment::createEnvironment()
/* END CONVERSION */;
            if (rc != SUCCESS) {
                throw SybaseException("Failed to allocate context", 1001);
            }
            
            // Initialize client library
            rc = /* CONVERTED FROM SYBASE: ct_init(ctx, CS_VERSION_125) */
true /* Oracle environment already initialized */
/* END CONVERSION */;
            if (rc != SUCCESS) {
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Failed to initialize client library", 1002);
            }
            
            // Set up error handlers using function pointers
            rc = ct_callback(ctx, NULL, CS_SET, CS_SERVERMSG_CB,
                           (CS_MSGFUNC_PTR)serverMsgHandler);
            if (rc != SUCCESS) {
                ct_exit(ctx, 0);
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Failed to set server message handler", 1003);
            }
            
            rc = ct_callback(ctx, NULL, CS_SET, CS_CLIENTMSG_CB,
                           (CS_MSGFUNC_PTR)clientMsgHandler);
            if (rc != SUCCESS) {
                ct_exit(ctx, 0);
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Failed to set client message handler", 1004);
            }
            
            // Allocate connection
            rc = /* CONVERTED FROM SYBASE: ct_con_alloc(ctx, &conn) */
conn = ctx->createConnection(username, password, connectionString)
/* END CONVERSION */;
            if (rc != SUCCESS) {
                ct_exit(ctx, 0);
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Failed to allocate connection", 1005);
            }
            
            // Set login credentials
            rc = ct_con_props(conn, CS_SET, CS_USERNAME, m_pUsername, -1, NULL);
            if (rc != SUCCESS) {
                /* CONVERTED FROM SYBASE: ct_con_drop(conn) */
Environment::terminateConnection(conn)
/* END CONVERSION */;
                ct_exit(ctx, 0);
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Failed to set username", 1006);
            }
            
            rc = ct_con_props(conn, CS_SET, CS_PASSWORD, m_pPassword, -1, NULL);
            if (rc != SUCCESS) {
                /* CONVERTED FROM SYBASE: ct_con_drop(conn) */
Environment::terminateConnection(conn)
/* END CONVERSION */;
                ct_exit(ctx, 0);
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Failed to set password", 1007);
            }
            
            // Connect to server
            rc = ct_connect(conn, m_pServer, -1);
            if (rc != SUCCESS) {
                /* CONVERTED FROM SYBASE: ct_con_drop(conn) */
Environment::terminateConnection(conn)
/* END CONVERSION */;
                ct_exit(ctx, 0);
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Failed to connect to server", 1008);
            }
            
            // Switch to the specified database
            char* useDbCmd = new char[strlen(m_pDatabase) + 5];  // "USE " + dbname + null
            sprintf(useDbCmd, "USE %s", m_pDatabase);
            
            rc = /* CONVERTED FROM SYBASE: ct_cmd_alloc(conn, &cmd) */
cmd = conn->createStatement()
/* END CONVERSION */;
            if (rc != SUCCESS) {
                delete[] useDbCmd;
                ct_close(conn, 0);
                /* CONVERTED FROM SYBASE: ct_con_drop(conn) */
Environment::terminateConnection(conn)
/* END CONVERSION */;
                ct_exit(ctx, 0);
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Failed to allocate command", 1009);
            }
            
            rc = ct_command(cmd, CS_LANG_CMD, useDbCmd, -1, 0);
            if (rc != SUCCESS) {
                delete[] useDbCmd;
                ct_cmd_drop(cmd);
                ct_close(conn, 0);
                /* CONVERTED FROM SYBASE: ct_con_drop(conn) */
Environment::terminateConnection(conn)
/* END CONVERSION */;
                ct_exit(ctx, 0);
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Failed to set command", 1010);
            }
            
            rc = /* CONVERTED FROM SYBASE: ct_send(cmd) */
cmd->execute()
/* END CONVERSION */;
            if (rc != SUCCESS) {
                delete[] useDbCmd;
                ct_cmd_drop(cmd);
                ct_close(conn, 0);
                /* CONVERTED FROM SYBASE: ct_con_drop(conn) */
Environment::terminateConnection(conn)
/* END CONVERSION */;
                ct_exit(ctx, 0);
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Failed to send command", 1011);
            }
            
            delete[] useDbCmd;
            
            // Process results
            int result_type;
            while ((rc = /* CONVERTED FROM SYBASE: ct_results(cmd, &result_type) */
result_type = (cmd->getResultSet() != NULL)
/* END CONVERSION */) == SUCCESS) {
                // Just process all results
            }
            
            if (rc != CS_END_RESULTS) {
                ct_cmd_drop(cmd);
                ct_close(conn, 0);
                /* CONVERTED FROM SYBASE: ct_con_drop(conn) */
Environment::terminateConnection(conn)
/* END CONVERSION */;
                ct_exit(ctx, 0);
                /* CONVERTED FROM SYBASE: cs_ctx_drop(ctx) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
                throw SybaseException("Error processing results", 1012);
            }
            
            // Store the connections in smart pointers
            m_pContext.reset(ctx);
            m_pConnection.reset(conn);
            
            // Drop the command, we'll create new ones as needed
            ct_cmd_drop(cmd);
            
            // Set the connected flag using bitwise OR
            m_flags |= CONNECTED;
            
            return true;
        }
        catch (SybaseException& ex) {
            // Log the error using the error callback if set
            if (m_pfnErrorCallback) {
                m_pfnErrorCallback(ex.what(), ex.getCode());
            }
            return false;
        }
    }
    
    // Disconnect with bitwise operations
    bool disconnect() {
        // Check if connected using bitwise AND
        if (!(m_flags & CONNECTED)) {
            return true;  // Already disconnected
        }
        
        // Reset all smart pointers to trigger the custom deleters
        m_pCommand.reset();
        m_pConnection.reset();
        m_pContext.reset();
        
        // Clear the connected flag using bitwise AND with NOT
        m_flags &= ~CONNECTED;
        
        return true;
    }
    
    // Create customer using template parameter binding
    template<typename... Args>
    bool createCustomer(int id, const char* name, const char* address, double balance) {
        if (!(m_flags & CONNECTED)) {
            logError("Not connected to database", 2001);
            return false;
        }
        
        try {
            // Create a smart pointer for the command
            CommandPtr cmd;
            Statement** rawCmd;
            
            // Allocate command
            if (/* CONVERTED FROM SYBASE: ct_cmd_alloc(m_pConnection.get(), &rawCmd) */
rawCmd = m_pConnection.get()->createStatement()
/* END CONVERSION */ != SUCCESS) {
                throw SybaseException("Failed to allocate command", 2002);
            }
            
            cmd.reset(rawCmd);
            
            // Prepare the SQL with placeholders
            const char* sql = "INSERT INTO Customers (ID, Name, Address, Balance) VALUES (?, ?, ?, ?)";
            
            if (ct_command(cmd.get(), CS_LANG_CMD, const_cast<char*>(sql), -1, 0) != SUCCESS) {
                throw SybaseException("Failed to set command", 2003);
            }
            
            // Using template specialization for type-safe parameter binding
            bindParam<int>(cmd.get(), 1, id);
            bindParam<char*>(cmd.get(), 2, const_cast<char*>(name));
            bindParam<char*>(cmd.get(), 3, const_cast<char*>(address));
            bindParam<double>(cmd.get(), 4, balance);
            
            // Execute the command
            if (/* CONVERTED FROM SYBASE: ct_send(cmd.get() */
cmd.get(->execute()
/* END CONVERSION */) != SUCCESS) {
                throw SybaseException("Failed to send command", 2004);
            }
            
            // Process results with pointer manipulation
            int resultType;
            int* pRowCount = new int(0);
            Status rc;
            
            while ((rc = /* CONVERTED FROM SYBASE: ct_results(cmd.get(), &resultType) */
resultType = (cmd.get()->getResultSet() != NULL)
/* END CONVERSION */) == SUCCESS) {
                if (resultType == CS_CMD_SUCCEED) {
                    // Get rows affected
                    ct_res_info(cmd.get(), CS_ROW_COUNT, pRowCount, 0, NULL);
                }
                else if (resultType == CS_CMD_FAIL) {
                    delete pRowCount;
                    throw SybaseException("Command failed", 2005);
                }
            }
            
            bool success = (*pRowCount > 0);
            
            // Cleanup
            delete pRowCount;
            
            return success;
        }
        catch (SybaseException& ex) {
            logError(ex.what(), ex.getCode());
            return false;
        }
    }
    
    // Read customer with complex pointer handling and result binding
    bool readCustomer(int id, Customer** ppCustomer) {
        if (!(m_flags & CONNECTED)) {
            logError("Not connected to database", 3001);
            return false;
        }
        
        // Verify output pointer is valid
        if (!ppCustomer) {
            logError("Invalid output parameter", 3002);
            return false;
        }
        
        try {
            // Create a smart pointer for the command
            CommandPtr cmd;
            Statement** rawCmd;
            
            // Allocate command
            if (/* CONVERTED FROM SYBASE: ct_cmd_alloc(m_pConnection.get(), &rawCmd) */
rawCmd = m_pConnection.get()->createStatement()
/* END CONVERSION */ != SUCCESS) {
                throw SybaseException("Failed to allocate command", 3003);
            }
            
            cmd.reset(rawCmd);
            
            // Prepare the SQL
            const char* sql = "SELECT ID, Name, Address, Balance FROM Customers WHERE ID = ?";
            
            if (ct_command(cmd.get(), CS_LANG_CMD, const_cast<char*>(sql), -1, 0) != SUCCESS) {
                throw SybaseException("Failed to set command", 3004);
            }
            
            // Bind parameter for ID
            bindParam<int>(cmd.get(), 1, id);
            
            // Execute the command
            if (/* CONVERTED FROM SYBASE: ct_send(cmd.get() */
cmd.get(->execute()
/* END CONVERSION */) != SUCCESS) {
                throw SybaseException("Failed to send command", 3005);
            }
            
            // Process results
            int resultType;
            Status rc;
            bool found = false;
            
            // Pre-allocate result buffers using raw pointers
            int* pResId = new int;
            char* pResName = new char[MAX_NAME_LENGTH + 1];
            char* pResAddress = new char[MAX_ADDRESS_LENGTH + 1];
            float* pResBalance = new float;
            
            // Clear memory for safety
            memset(pResName, 0, MAX_NAME_LENGTH + 1);
            memset(pResAddress, 0, MAX_ADDRESS_LENGTH + 1);
            
            while ((rc = /* CONVERTED FROM SYBASE: ct_results(cmd.get(), &resultType) */
resultType = (cmd.get()->getResultSet() != NULL)
/* END CONVERSION */) == SUCCESS) {
                if (resultType == CS_ROW_RESULT) {
                    // Bind result columns using template helper
                    bindResult<int>(cmd.get(), 1, pResId, sizeof(int));
                    bindResult<char>(cmd.get(), 2, pResName, MAX_NAME_LENGTH);
                    bindResult<char>(cmd.get(), 3, pResAddress, MAX_ADDRESS_LENGTH);
                    bindResult<float>(cmd.get(), 4, pResBalance, sizeof(float));
                    
                    // Fetch the data
                    if (/* CONVERTED FROM SYBASE: ct_fetch(cmd.get(), 0, 0, 0) */
cmd.get()->getResultSet()->next()
/* END CONVERSION */ == SUCCESS) {
                        // Allocate customer object
                        *ppCustomer = new Customer();
                        
                        // Set data via pointers
                        (*ppCustomer)->id = *pResId;
                        (*ppCustomer)->setName(reinterpret_cast<char*>(pResName));
                        (*ppCustomer)->setAddress(reinterpret_cast<char*>(pResAddress));
                        (*ppCustomer)->balance = static_cast<double>(*pResBalance);
                        
                        found = true;
                    }
                    
                    // Fetch any remaining rows (should be only one)
                    while (/* CONVERTED FROM SYBASE: ct_fetch(cmd.get(), 0, 0, 0) */
cmd.get()->getResultSet()->next()
/* END CONVERSION */ == SUCCESS) {
                        // Ignore additional rows
                    }
                }
            }
            
            // Clean up result pointers
            delete pResId;
            delete[] pResName;
            delete[] pResAddress;
            delete pResBalance;
            
            if (rc != CS_END_RESULTS) {
                throw SybaseException("Error processing results", 3006);
            }
            
            return found;
        }
        catch (SybaseException& ex) {
            logError(ex.what(), ex.getCode());
            return false;
        }
    }
    
    // Delete customer with bit flag checking
    bool deleteCustomer(int id) {
        if (!(m_flags & CONNECTED)) {
            logError("Not connected to database", 4001);
            return false;
        }
        
        try {
            // Create a smart pointer for the command
            CommandPtr cmd;
            Statement** rawCmd;
            
            // Allocate command
            if (/* CONVERTED FROM SYBASE: ct_cmd_alloc(m_pConnection.get(), &rawCmd) */
rawCmd = m_pConnection.get()->createStatement()
/* END CONVERSION */ != SUCCESS) {
                throw SybaseException("Failed to allocate command", 4002);
            }
            
            cmd.reset(rawCmd);
            
            // Prepare the SQL
            const char* sql = "DELETE FROM Customers WHERE ID = ?";
            
            if (ct_command(cmd.get(), CS_LANG_CMD, const_cast<char*>(sql), -1, 0) != SUCCESS) {
                throw SybaseException("Failed to set command", 4003);
            }
            
            // Bind parameter for ID using templated function
            bindParam<int>(cmd.get(), 1, id);
            
            // Execute the command
            if (/* CONVERTED FROM SYBASE: ct_send(cmd.get() */
cmd.get(->execute()
/* END CONVERSION */) != SUCCESS) {
                throw SybaseException("Failed to send command", 4004);
            }
            
            // Process results with raw pointer
            int resultType;
            int* pRowCount = new int(0);
            Status rc;
            
            while ((rc = /* CONVERTED FROM SYBASE: ct_results(cmd.get(), &resultType) */
resultType = (cmd.get()->getResultSet() != NULL)
/* END CONVERSION */) == SUCCESS) {
                if (resultType == CS_CMD_SUCCEED) {
                    // Get rows affected
                    ct_res_info(cmd.get(), CS_ROW_COUNT, pRowCount, 0, NULL);
                }
                else if (resultType == CS_CMD_FAIL) {
                    delete pRowCount;
                    throw SybaseException("Command failed", 4005);
                }
            }
            
            bool success = (*pRowCount > 0);
            
            // Cleanup
            delete pRowCount;
            
            return success;
        }
        catch (SybaseException& ex) {
            logError(ex.what(), ex.getCode());
            return false;
        }
    }
    
    // Set error callback using std::function
    void setErrorCallback(std::function<void(const char*, int)> callback) {
        m_pfnErrorCallback = callback;
    }
    
    // Enable/disable debug mode using bit flags
    void setDebugMode(bool enable) {
        if (enable) {
            m_flags |= DEBUG_MODE;
        } else {
            m_flags &= ~DEBUG_MODE;
        }
    }

private:
    // Template function for binding parameters with type safety
    template<typename T>
    void bindParam(Statement** cmd, int paramNum, T value) {
        static_assert(std::is_same<T, int>::value || 
                     std::is_same<T, float>::value ||
                     std::is_same<T, double>::value ||
                     std::is_same<T, char*>::value,
                     "Unsupported parameter type");
        
        Type paramFmt;
        memset(&paramFmt, 0, sizeof(paramFmt));
        
        // Use template specialization for type-specific settings
        paramFmt.datatype = SybaseTypeTraits<T>::datatype;
        
        // Special handling for strings
        if constexpr (std::is_same<T, char*>::value) {
            paramFmt.maxlength = strlen(value);
        } else {
            paramFmt.maxlength = SybaseTypeTraits<T>::buffer_size;
        }
        
        paramFmt.status = CS_INPUTVALUE;
        paramFmt.locale = NULL;
        
        // Handle float/double conversion
        if constexpr (std::is_same<T, double>::value) {
            float floatVal = static_cast<float>(value);
            if (ct_param(cmd, &paramFmt, &floatVal, sizeof(floatVal), 0) != SUCCESS) {
                throw SybaseException("Failed to bind parameter", 5001);
            }
        } else {
            // All other types
            if (ct_param(cmd, &paramFmt, const_cast<void*>(static_cast<const void*>(&value)),
                        paramFmt.maxlength, 0) != SUCCESS) {
                throw SybaseException("Failed to bind parameter", 5002);
            }
        }
    }
    
    // Template function for binding result columns with type safety
    template<typename T>
    void bindResult(Statement** cmd, int colNum, T* buffer, int maxLen) {
        Type resFmt;
        memset(&resFmt, 0, sizeof(resFmt));
        
        // Set column properties based on type
        if constexpr (std::is_same<T, int>::value) {
            resFmt.datatype = CS_INT_TYPE;
            resFmt.format = CS_FMT_UNUSED;
        } else if constexpr (std::is_same<T, float>::value) {
            resFmt.datatype = CS_FLOAT_TYPE;
            resFmt.format = CS_FMT_UNUSED;
        } else if constexpr (std::is_same<T, char>::value) {
            resFmt.datatype = CS_CHAR_TYPE;
            resFmt.format = CS_FMT_NULLTERM;
        } else {
            // This will cause a compile error for unsupported types
            static_assert(sizeof(T) == 0, "Unsupported result type");
        }
        
        resFmt.maxlength = maxLen;
        resFmt.count = 1;
        
        if (ct_bind(cmd, colNum, &resFmt, buffer, NULL, NULL) != SUCCESS) {
            throw SybaseException("Failed to bind result column", 6001);
        }
    }
    
    // Log error with optional error callback
    void logError(const char* message, int code) {
        if (m_pfnErrorCallback) {
            m_pfnErrorCallback(message, code);
        }
        
        // Also log to console if in debug mode
        if (m_flags & DEBUG_MODE) {
            fprintf(stderr, "ERROR %d: %s\n", code, message);
        }
    }
    
    // Static message handlers - using C-style callbacks
    static Status serverMsgHandler(Environment** context, Connection** connection, CS_SERVERMSG* message) {
        fprintf(stderr, "Server Message: %s (Error: %d, Severity: %d)\n", 
                message->text, message->msgnumber, message->severity);
        return SUCCESS;
    }
    
    static Status clientMsgHandler(Environment** context, Connection** connection, CS_CLIENTMSG* message) {
        fprintf(stderr, "Client Message: %s (Error: %d, Severity: %d)\n", 
                message->msgstring, message->msgnumber, CS_SEVERITY(message->severity));
        return SUCCESS;
    }
};

/**
 * @brief Example usage - main function demonstration
 */
int main() {
    // Create manager with raw C-style strings
    SybaseManager* pManager = SybaseManager::create(
        "SYBASESERVER", "sa", "p@$$w0rd", "Customers");
    
    // Set debug mode with bit flags
    pManager->setDebugMode(true);
    
    // Set a lambda as error callback
    pManager->setErrorCallback([](const char* msg, int code) {
        printf("DB ERROR [%d]: %s\n", code, msg);
    });
    
    // Connect to database
    if (!pManager->connect()) {
        fprintf(stderr, "Failed to connect to database\n");
        delete pManager;  // Manual memory management
        return 1;
    }
    
    // Create sample customer
    if (pManager->createCustomer(1001, "John Smith", "123 Tech St", 5000.75)) {
        printf("Customer created successfully\n");
    }
    
    // Read customer with double-indirection pointer
    Customer* pCustomer = nullptr;
    if (pManager->readCustomer(1001, &pCustomer)) {
        printf("Customer found: ID=%d, Name=%s, Balance=%.2f\n", 
               pCustomer->id, pCustomer->getName(), pCustomer->balance);
        delete pCustomer;  // Clean up allocated memory
    }
    
    // Delete customer
    if (pManager->deleteCustomer(1001)) {
        printf("Customer deleted successfully\n");
    }
    
    // Disconnect and clean up
    pManager->disconnect();
    delete pManager;  // Manual memory management
    
    return 0;
}
