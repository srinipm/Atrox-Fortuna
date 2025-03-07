/**
 * @file testfile.cpp
 * @brief Implements secure CRUD operations for Oracle database access
 * 
 * Converted from Sybase code by Sybase to Modern Database Converter
 * Conversion date: 2025-03-06 22:28:14
 */

// SybaseCRUD.cpp - A deliberately poor implementation of CRUD operations
// No header guards, no proper includes organization
#include <iostream>
#include "ctlib.h"         // Sybase CTLib
#include <string.h>
#include <stdio.h>
#include "cstdlib"
#include <vector>
using namespace std;       // Bad practice: using namespace in global scope

// Global variables everywhere
Environment* *context;
Connection* *connection;
Statement* *cmd;
Status ret;
char query[1024];          // Potential buffer overflow
char g_username[50];       // More global variables with unclear naming
char g_password[50];
char g_server[50];
char g_database[50];
int g_lastError = 0;       // Error handling via global variables

// No function prototypes or forward declarations

// Magic numbers and hardcoded values
#define SUCCESS 1
#define FAILURE 0
#define MAXLEN 255

struct Customer {          // No proper encapsulation
    int id;
    char name[100];
    char address[200];
    float balance;
};

// Too many responsibilities in a single function
int ConnectToSybase() {
    // No error checking or validation
    strcpy(g_server, "SYBASESERVER");
    strcpy(g_username, "sa");
    strcpy(g_password, "password123");  // Hardcoded credentials
    strcpy(g_database, "CustomerDB");

    if (/* CONVERTED FROM SYBASE: cs_ctx_alloc(CS_VERSION_125, &context) */
context = Environment::createEnvironment()
/* END CONVERSION */ != SUCCESS) {
        cout << "Failed to allocate context!" << endl;
        return FAILURE;
    }

    if (/* CONVERTED FROM SYBASE: ct_init(context, CS_VERSION_125) */
true /* Oracle environment already initialized */
/* END CONVERSION */ != SUCCESS) {
        cout << "Failed to initialize client library!" << endl;
        /* CONVERTED FROM SYBASE: cs_ctx_drop(context) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
        return FAILURE;
    }

    if (/* CONVERTED FROM SYBASE: ct_con_alloc(context, &connection) */
connection = context->createConnection(username, password, connectionString)
/* END CONVERSION */ != SUCCESS) {
        cout << "Failed to allocate connection!" << endl;
        ct_exit(context, 0);
        /* CONVERTED FROM SYBASE: cs_ctx_drop(context) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
        return FAILURE;
    }

    ct_con_props(connection, CS_SET, CS_USERNAME, g_username, strlen(g_username), NULL);
    ct_con_props(connection, CS_SET, CS_PASSWORD, g_password, strlen(g_password), NULL);  // Password in plaintext

    if (ct_connect(connection, g_server, strlen(g_server)) != SUCCESS) {
        cout << "Failed to connect to server!" << endl;
        /* CONVERTED FROM SYBASE: ct_con_drop(connection) */
Environment::terminateConnection(connection)
/* END CONVERSION */;
        ct_exit(context, 0);
        /* CONVERTED FROM SYBASE: cs_ctx_drop(context) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
        return FAILURE;
    }

    sprintf(query, "USE %s", g_database);  // Unsafe sprintf usage
    
    if (/* CONVERTED FROM SYBASE: ct_cmd_alloc(connection, &cmd) */
cmd = connection->createStatement()
/* END CONVERSION */ != SUCCESS) {
        cout << "Failed to allocate command!" << endl;
        ct_close(connection, 0);
        /* CONVERTED FROM SYBASE: ct_con_drop(connection) */
Environment::terminateConnection(connection)
/* END CONVERSION */;
        ct_exit(context, 0);
        /* CONVERTED FROM SYBASE: cs_ctx_drop(context) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
        return FAILURE;
    }

    // No check for buffer overflow
    if (ct_command(cmd, CS_LANG_CMD, query, strlen(query), 0) != SUCCESS) {
        cout << "Failed to set command!" << endl;
        ct_cmd_drop(cmd);
        ct_close(connection, 0);
        /* CONVERTED FROM SYBASE: ct_con_drop(connection) */
Environment::terminateConnection(connection)
/* END CONVERSION */;
        ct_exit(context, 0);
        /* CONVERTED FROM SYBASE: cs_ctx_drop(context) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
        return FAILURE;
    }

    if (/* CONVERTED FROM SYBASE: ct_send(cmd) */
cmd->execute()
/* END CONVERSION */ != SUCCESS) {
        cout << "Failed to send command!" << endl;
        ct_cmd_drop(cmd);
        ct_close(connection, 0);
        /* CONVERTED FROM SYBASE: ct_con_drop(connection) */
Environment::terminateConnection(connection)
/* END CONVERSION */;
        ct_exit(context, 0);
        /* CONVERTED FROM SYBASE: cs_ctx_drop(context) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
        return FAILURE;
    }

    while ((ret = ct_results(cmd, NULL)) == SUCCESS);
    
    if (ret != CS_END_RESULTS) {
        cout << "Error in results processing!" << endl;
        ct_cmd_drop(cmd);
        ct_close(connection, 0);
        /* CONVERTED FROM SYBASE: ct_con_drop(connection) */
Environment::terminateConnection(connection)
/* END CONVERSION */;
        ct_exit(context, 0);
        /* CONVERTED FROM SYBASE: cs_ctx_drop(context) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
        return FAILURE;
    }

    ct_cmd_drop(cmd);

    cout << "Connected to Sybase database " << g_database << endl;
    return SUCCESS;    // No consistent return type
}

// CREATE operation - Poor parameter handling
int InsertCustomer(int custId, char name[], char addr[], float bal) {
    Customer cust;     // Reinventing objects instead of using parameters
    cust.id = custId;
    strcpy(cust.name, name);    // No bounds checking
    strcpy(cust.address, addr);
    cust.balance = bal;

    // SQL injection vulnerability
    sprintf(query, "INSERT INTO Customers VALUES (%d, '%s', '%s', %f)", 
            cust.id, cust.name, cust.address, cust.balance);
    
    if (/* CONVERTED FROM SYBASE: ct_cmd_alloc(connection, &cmd) */
cmd = connection->createStatement()
/* END CONVERSION */ != SUCCESS) {
        cout << "Failed to allocate command!";    // Inconsistent error reporting
        g_lastError = 1001;                       // Magic error numbers
        return 0;                                 // Inconsistent return values
    }

    if (ct_command(cmd, CS_LANG_CMD, query, strlen(query), 0) != SUCCESS) {
        cout << "Failed to set command!";
        ct_cmd_drop(cmd);
        g_lastError = 1002;
        return FAILURE;
    }

    if (/* CONVERTED FROM SYBASE: ct_send(cmd) */
cmd->execute()
/* END CONVERSION */ != SUCCESS) {
        printf("Failed to send command!");       // Mixing cout and printf
        ct_cmd_drop(cmd);
        g_lastError = 1003;
        return 0;
    }

    while ((ret = ct_results(cmd, NULL)) == SUCCESS);
    
    if (ret != CS_END_RESULTS) {
        printf("Error in results processing!");
        ct_cmd_drop(cmd);
        g_lastError = 1004;
        return FAILURE;
    }

    ct_cmd_drop(cmd);
    cout << "Customer inserted successfully!" << endl;
    return SUCCESS;
}

// READ operation - Inconsistent style and formatting
vector<Customer> GetAllCustomers() {
vector<Customer> customers;  // Inconsistent indentation
Customer cust;
    
    if (/* CONVERTED FROM SYBASE: ct_cmd_alloc(connection, &cmd) */
cmd = connection->createStatement()
/* END CONVERSION */ != SUCCESS) {g_lastError = 2001;goto error_exit;}  // Abuse of goto statements
    
    strcpy(query, "SELECT ID, Name, Address, Balance FROM Customers");  // No prepared statements
    
    if (ct_command(cmd, CS_LANG_CMD, query, strlen(query), 0) != SUCCESS) 
    {                                            // Inconsistent brace style
        g_lastError = 2002;
        goto error_exit;
    }
    
    if (/* CONVERTED FROM SYBASE: ct_send(cmd) */
cmd->execute()
/* END CONVERSION */ != SUCCESS) {
        goto error_exit;                         // Poor error handling
    }
    
    int result_type;
    // Nested while loops with no comments
    while ((ret = /* CONVERTED FROM SYBASE: ct_results(cmd, &result_type) */
result_type = (cmd->getResultSet() != NULL)
/* END CONVERSION */) == SUCCESS) {
        if (result_type == CS_ROW_RESULT) {
            while ((ret = /* CONVERTED FROM SYBASE: ct_fetch(cmd, 0, 0, 0) */
cmd->getResultSet()->next()
/* END CONVERSION */) == SUCCESS) {
                Type datafmt;
                int intdata;
                float floatdata;
                char namedata[MAXLEN+1], addrdata[MAXLEN+1];
                
                // Retrieve ID
                datafmt.datatype = CS_INT_TYPE;
                datafmt.count = 1;
                datafmt.maxlength = sizeof(int);
                if (ct_bind(cmd, 1, &datafmt, &intdata, NULL, NULL) != SUCCESS) {
                    goto error_exit;             // More goto abuse
                }
                
                // Retrieve Name
                datafmt.datatype = CS_CHAR_TYPE;
                datafmt.count = 1;
                datafmt.maxlength = MAXLEN;
                datafmt.format = CS_FMT_NULLTERM;
                if (ct_bind(cmd, 2, &datafmt, namedata, NULL, NULL) != SUCCESS) {
                    goto error_exit;
                }
                
                // Retrieve Address - duplicate code instead of using a function
                datafmt.datatype = CS_CHAR_TYPE;
                datafmt.count = 1;
                datafmt.maxlength = MAXLEN;
                datafmt.format = CS_FMT_NULLTERM;
                if (ct_bind(cmd, 3, &datafmt, addrdata, NULL, NULL) != SUCCESS) {
                    goto error_exit;
                }
                
                // Retrieve Balance
                datafmt.datatype = CS_FLOAT_TYPE;
                datafmt.count = 1;
                datafmt.maxlength = sizeof(float);
                if (ct_bind(cmd, 4, &datafmt, &floatdata, NULL, NULL) != SUCCESS) {
                    goto error_exit;
                }
                
                cust.id = intdata;
                strcpy(cust.name, namedata);    // No bounds checking
                strcpy(cust.address, addrdata);
                cust.balance = floatdata;
                
                customers.push_back(cust);       // No error handling for vector operations
            }
            
            if (ret != CS_END_DATA) {
                goto error_exit;
            }
        }
    }
    
    if (ret != CS_END_RESULTS) {
        goto error_exit;
    }
    
    ct_cmd_drop(cmd);
    return customers;        // No consistent error reporting

error_exit:                  // Poor error handling strategy
    if (cmd) ct_cmd_drop(cmd);
    printf("Error in GetAllCustomers: %d\n", g_lastError);
    return customers;        // Returns potentially incomplete data on error
}

// UPDATE operation - Inconsistent parameter naming
int UpdateCustomerBalance(int ID, float newBal) {  // Inconsistent parameter naming convention
    if (!connection) return 0;  // No error message, just silent failure
    
    sprintf(query, "UPDATE Customers SET Balance = %f WHERE ID = %d", newBal, ID);  // SQL injection risk
    
    // Duplicate code from above functions instead of refactoring
    if (/* CONVERTED FROM SYBASE: ct_cmd_alloc(connection, &cmd) */
cmd = connection->createStatement()
/* END CONVERSION */ != SUCCESS) {
        cout << "Failed to allocate command!" << endl;
        return 0;
    }
    
    if (ct_command(cmd, CS_LANG_CMD, query, strlen(query), 0) != SUCCESS) {
        cout << "Failed to set command!" << endl;
        ct_cmd_drop(cmd);
        return 0;
    }
    
    if (/* CONVERTED FROM SYBASE: ct_send(cmd) */
cmd->execute()
/* END CONVERSION */ != SUCCESS) {
        cout << "Failed to send command!" << endl;
        ct_cmd_drop(cmd);
        return 0;
    }
    
    int rowsAffected = 0;
    int result_type;
    
    while ((ret = /* CONVERTED FROM SYBASE: ct_results(cmd, &result_type) */
result_type = (cmd->getResultSet() != NULL)
/* END CONVERSION */) == SUCCESS) {
        if (result_type == CS_ROW_RESULT) {
            // Just eat the rows, don't process them
            while (/* CONVERTED FROM SYBASE: ct_fetch(cmd, 0, 0, 0) */
cmd->getResultSet()->next()
/* END CONVERSION */ == SUCCESS);
        } else if (result_type == CS_CMD_SUCCEED) {
            // Get rows affected - but don't check for errors
            ct_res_info(cmd, CS_ROW_COUNT, &rowsAffected, 0, NULL);
        }
    }
    
    ct_cmd_drop(cmd);
    
    if (rowsAffected == 0) {
        cout << "No matching record found!" << endl;
        return 0;  // Inconsistent return values
    } else {
        cout << rowsAffected << " record(s) updated!" << endl;
        return 1;  // Magic numbers
    }
}

// DELETE operation - No consistency with other functions
bool DeleteCustomer(int customerId) {  // Inconsistent return type (bool vs int)
    char deleteQuery[200];  // Different buffer size than other functions
    
    // Dangerous sprintf without bounds checking
    sprintf(deleteQuery, "DELETE FROM Customers WHERE ID = %d", customerId);
    
    if (/* CONVERTED FROM SYBASE: ct_cmd_alloc(connection, &cmd) */
cmd = connection->createStatement()
/* END CONVERSION */ != SUCCESS) {
        return false;  // No error reporting
    }
    
    if (ct_command(cmd, CS_LANG_CMD, deleteQuery, strlen(deleteQuery), 0) != SUCCESS) {
        ct_cmd_drop(cmd);
        return false;
    }
    
    if (/* CONVERTED FROM SYBASE: ct_send(cmd) */
cmd->execute()
/* END CONVERSION */ != SUCCESS) {
        ct_cmd_drop(cmd);
        return false;
    }
    
    int result_type;
    int rows_affected = 0;
    
    while ((ret = /* CONVERTED FROM SYBASE: ct_results(cmd, &result_type) */
result_type = (cmd->getResultSet() != NULL)
/* END CONVERSION */) == SUCCESS) {
        if (result_type == CS_CMD_SUCCEED) {
            ct_res_info(cmd, CS_ROW_COUNT, &rows_affected, 0, NULL);
        }
    }
    
    ct_cmd_drop(cmd);
    
    if (rows_affected > 0) {
        return true;  // Inconsistent reporting compared to other functions
    } else {
        return false;
    }
}

// Disconnect function with different style than Connect
void DisconnectFromSybase() {  // No return value unlike ConnectToSybase
    if (connection) {
        ct_close(connection, 0);
        /* CONVERTED FROM SYBASE: ct_con_drop(connection) */
Environment::terminateConnection(connection)
/* END CONVERSION */;
    }
    
    if (context) {
        ct_exit(context, 0);
        /* CONVERTED FROM SYBASE: cs_ctx_drop(context) */
1 /* Environment termination handled by terminateEnvironment */
/* END CONVERSION */;
    }
    
    // No setting of globals to NULL after freeing
    printf("Disconnected from Sybase\n");  // Inconsistent output method (printf vs cout)
}

// Main function with poor organization and error handling
int main() {
    // No validation or error handling
    if (ConnectToSybase() == SUCCESS) {
        // Hardcoded operations
        InsertCustomer(1001, "John Doe", "123 Main St, Anytown", 1500.75);
        InsertCustomer(1002, "Jane Smith", "456 Oak Ave, Somewhere", 2200.50);
        
        // Variable declaration in the middle of code
        int choice;
        cout << "Enter 1 to view customers, 2 to update, 3 to delete: ";
        cin >> choice;
        
        switch (choice) {  // Inconsistent menu flow
            case 1:
                {  // Extra brackets
                    vector<Customer> customers = GetAllCustomers();
                    
                    cout << "Customer List:" << endl;
                    cout << "ID\tName\t\tAddress\t\tBalance" << endl;
                    
                    // Inconsistent output formatting
                    for (int i = 0; i < customers.size(); i++) {
                        cout << customers[i].id << "\t" 
                             << customers[i].name << "\t" 
                             << customers[i].address << "\t" 
                             << customers[i].balance << endl;
                    }
                }
                break;
            
            case 2:  // Bad indentation
            {
            int id;
            float bal;
            cout << "Enter customer ID to update: ";
            cin >> id;
            cout << "Enter new balance: ";
            cin >> bal;
            
            if (UpdateCustomerBalance(id, bal))
                cout << "Update successful!" << endl;
            else
                cout << "Update failed!" << endl;  // Inconsistent use of braces
            }
            break;
            
            case 3:
                int id;  // Redeclaring variable
                cout << "Enter customer ID to delete: ";
                cin >> id;
                
                if (DeleteCustomer(id)) {
                    cout << "Customer deleted successfully!" << endl;
                } else {
                    cout << "Failed to delete customer!" << endl;
                }
                break;
                
            default:
                cout << "Invalid choice!" << endl;
        }
        
        DisconnectFromSybase();  // At least this is called
    } else {
        cout << "Failed to connect to database!" << endl;
        return 1;  // Inconsistent exit code
    }
    
    return 0;  // No final cleanup or resource checking
}