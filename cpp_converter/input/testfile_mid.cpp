// SybaseOperations.cpp
// This is a program to do database operations on Sybase
// Written by: John Smith, Junior Programmer
// Date: 2/12/2005
// Company: IT Services Corp.

#include <iostream>
#include <stdio.h>
#include <string.h>
#include "ctlib.h"  // Sybase CT-Lib 

// Global variables to use throughout the program
CS_CONTEXT *g_context;
CS_CONNECTION *g_connection;
CS_COMMAND *g_command;
CS_RETCODE g_retcode;
char g_query[2000];  // Big buffer for queries

// Login information
char server[50] = "SYBSERVER1";
char username[50] = "sa";
char password[50] = "sa_password123";
char database[50] = "CustomerDB";

// Error flag
int error = 0;

// Structure for customer data
struct Customer {
    int id;
    char name[100];
    char address[200];
    double balance;
};

// Function to check if there was an error
void check_error() {
    if (error != 0) {
        printf("There was an error! Error code: %d\n", error);
        exit(1);  // Exit the program if there's an error
    }
}

// Function to connect to the database
void ConnectToDatabase() {
    printf("Connecting to database...\n");
    
    // Allocate a context
    g_retcode = cs_ctx_alloc(CS_VERSION_125, &g_context);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to allocate context!\n");
        error = 1;
        return;
    }
    
    // Initialize Client-Library
    g_retcode = ct_init(g_context, CS_VERSION_125);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to initialize Client-Library!\n");
        error = 2;
        return;
    }
    
    // Allocate a connection
    g_retcode = ct_con_alloc(g_context, &g_connection);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to allocate connection!\n");
        error = 3;
        return;
    }
    
    // Set the username
    g_retcode = ct_con_props(g_connection, CS_SET, CS_USERNAME, 
                           username, CS_NULLTERM, NULL);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to set username!\n");
        error = 4;
        return;
    }
    
    // Set the password
    g_retcode = ct_con_props(g_connection, CS_SET, CS_PASSWORD, 
                           password, CS_NULLTERM, NULL);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to set password!\n");
        error = 5;
        return;
    }
    
    // Connect to the server
    g_retcode = ct_connect(g_connection, server, CS_NULLTERM);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to connect to server!\n");
        error = 6;
        return;
    }
    
    // Switch to the specified database
    sprintf(g_query, "USE %s", database);
    
    g_retcode = ct_cmd_alloc(g_connection, &g_command);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to allocate command!\n");
        error = 7;
        return;
    }
    
    g_retcode = ct_command(g_command, CS_LANG_CMD, g_query, 
                         strlen(g_query), CS_UNUSED);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to initialize command!\n");
        error = 8;
        return;
    }
    
    g_retcode = ct_send(g_command);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to send command!\n");
        error = 9;
        return;
    }
    
    // Process all results
    CS_INT result_type;
    while ((g_retcode = ct_results(g_command, &result_type)) == CS_SUCCEED) {
        // Do nothing with the results
    }
    
    if (g_retcode != CS_END_RESULTS) {
        printf("Error in results processing!\n");
        error = 10;
        return;
    }
    
    // Drop the command
    ct_cmd_drop(g_command);
    
    printf("Connected to database successfully!\n");
}

// Function to create a customer
void AddCustomer(int id, char *name, char *address, double balance) {
    printf("Adding customer...\n");
    
    // Build the query - this is not safe for real use!
    sprintf(g_query, "INSERT INTO Customers VALUES (%d, '%s', '%s', %f)", 
           id, name, address, balance);
    
    // Allocate a command
    g_retcode = ct_cmd_alloc(g_connection, &g_command);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to allocate command!\n");
        error = 11;
        return;
    }
    
    // Initialize the command
    g_retcode = ct_command(g_command, CS_LANG_CMD, g_query, 
                         strlen(g_query), CS_UNUSED);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to initialize command!\n");
        error = 12;
        return;
    }
    
    // Send the command
    g_retcode = ct_send(g_command);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to send command!\n");
        error = 13;
        return;
    }
    
    // Process all results
    CS_INT result_type;
    while ((g_retcode = ct_results(g_command, &result_type)) == CS_SUCCEED) {
        // Do nothing with the results
    }
    
    if (g_retcode != CS_END_RESULTS) {
        printf("Error in results processing!\n");
        error = 14;
        return;
    }
    
    // Drop the command
    ct_cmd_drop(g_command);
    
    printf("Customer added successfully!\n");
}

// Function to get customer details
Customer GetCustomer(int id) {
    printf("Getting customer details...\n");
    
    Customer customer;
    customer.id = 0;  // Initialize with default values
    strcpy(customer.name, "");
    strcpy(customer.address, "");
    customer.balance = 0.0;
    
    // Build the query
    sprintf(g_query, "SELECT ID, Name, Address, Balance FROM Customers WHERE ID = %d", id);
    
    // Allocate a command
    g_retcode = ct_cmd_alloc(g_connection, &g_command);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to allocate command!\n");
        error = 15;
        return customer;
    }
    
    // Initialize the command
    g_retcode = ct_command(g_command, CS_LANG_CMD, g_query, 
                         strlen(g_query), CS_UNUSED);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to initialize command!\n");
        error = 16;
        return customer;
    }
    
    // Send the command
    g_retcode = ct_send(g_command);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to send command!\n");
        error = 17;
        return customer;
    }
    
    // Process the results
    CS_INT result_type;
    CS_INT id_data;
    CS_CHAR name_data[100];
    CS_CHAR address_data[200];
    CS_FLOAT balance_data;
    CS_DATAFMT id_fmt, name_fmt, address_fmt, balance_fmt;
    
    while ((g_retcode = ct_results(g_command, &result_type)) == CS_SUCCEED) {
        if (result_type == CS_ROW_RESULT) {
            // Set up the ID format
            memset(&id_fmt, 0, sizeof(id_fmt));
            id_fmt.datatype = CS_INT_TYPE;
            id_fmt.maxlength = sizeof(CS_INT);
            id_fmt.format = CS_FMT_UNUSED;
            id_fmt.count = 1;
            
            // Bind the ID
            g_retcode = ct_bind(g_command, 1, &id_fmt, &id_data, NULL, NULL);
            if (g_retcode != CS_SUCCEED) {
                printf("Failed to bind ID column!\n");
                error = 18;
                break;
            }
            
            // Set up the Name format
            memset(&name_fmt, 0, sizeof(name_fmt));
            name_fmt.datatype = CS_CHAR_TYPE;
            name_fmt.maxlength = sizeof(name_data);
            name_fmt.format = CS_FMT_NULLTERM;
            name_fmt.count = 1;
            
            // Bind the Name
            g_retcode = ct_bind(g_command, 2, &name_fmt, name_data, NULL, NULL);
            if (g_retcode != CS_SUCCEED) {
                printf("Failed to bind Name column!\n");
                error = 19;
                break;
            }
            
            // Set up the Address format
            memset(&address_fmt, 0, sizeof(address_fmt));
            address_fmt.datatype = CS_CHAR_TYPE;
            address_fmt.maxlength = sizeof(address_data);
            address_fmt.format = CS_FMT_NULLTERM;
            address_fmt.count = 1;
            
            // Bind the Address
            g_retcode = ct_bind(g_command, 3, &address_fmt, address_data, NULL, NULL);
            if (g_retcode != CS_SUCCEED) {
                printf("Failed to bind Address column!\n");
                error = 20;
                break;
            }
            
            // Set up the Balance format
            memset(&balance_fmt, 0, sizeof(balance_fmt));
            balance_fmt.datatype = CS_FLOAT_TYPE;
            balance_fmt.maxlength = sizeof(CS_FLOAT);
            balance_fmt.format = CS_FMT_UNUSED;
            balance_fmt.count = 1;
            
            // Bind the Balance
            g_retcode = ct_bind(g_command, 4, &balance_fmt, &balance_data, NULL, NULL);
            if (g_retcode != CS_SUCCEED) {
                printf("Failed to bind Balance column!\n");
                error = 21;
                break;
            }
            
            // Fetch the row
            if (ct_fetch(g_command, CS_UNUSED, CS_UNUSED, CS_UNUSED) == CS_SUCCEED) {
                // Copy the data to our customer struct
                customer.id = id_data;
                strcpy(customer.name, (char *)name_data);
                strcpy(customer.address, (char *)address_data);
                customer.balance = balance_data;
                
                printf("Customer found!\n");
            } else {
                printf("No customer found with ID %d\n", id);
            }
            
            // Fetch any remaining rows to clean up
            while (ct_fetch(g_command, CS_UNUSED, CS_UNUSED, CS_UNUSED) == CS_SUCCEED) {
                // Do nothing
            }
        }
    }
    
    // Drop the command
    ct_cmd_drop(g_command);
    
    return customer;
}

// Function to update customer balance
void UpdateCustomerBalance(int id, double new_balance) {
    printf("Updating customer balance...\n");
    
    // Build the query
    sprintf(g_query, "UPDATE Customers SET Balance = %f WHERE ID = %d", 
           new_balance, id);
    
    // Allocate a command
    g_retcode = ct_cmd_alloc(g_connection, &g_command);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to allocate command!\n");
        error = 22;
        return;
    }
    
    // Initialize the command
    g_retcode = ct_command(g_command, CS_LANG_CMD, g_query, 
                         strlen(g_query), CS_UNUSED);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to initialize command!\n");
        error = 23;
        return;
    }
    
    // Send the command
    g_retcode = ct_send(g_command);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to send command!\n");
        error = 24;
        return;
    }
    
    // Process all results
    CS_INT result_type;
    CS_INT rows_affected = 0;
    
    while ((g_retcode = ct_results(g_command, &result_type)) == CS_SUCCEED) {
        if (result_type == CS_CMD_SUCCEED) {
            // Get the number of rows affected
            g_retcode = ct_res_info(g_command, CS_ROW_COUNT, &rows_affected, 
                                  CS_UNUSED, NULL);
            if (g_retcode != CS_SUCCEED) {
                printf("Failed to get rows affected!\n");
                error = 25;
                break;
            }
        }
    }
    
    if (g_retcode != CS_END_RESULTS) {
        printf("Error in results processing!\n");
        error = 26;
        return;
    }
    
    // Drop the command
    ct_cmd_drop(g_command);
    
    if (rows_affected > 0) {
        printf("Customer balance updated successfully! Rows affected: %d\n", rows_affected);
    } else {
        printf("No customer found with ID %d\n", id);
    }
}

// Function to delete a customer
void DeleteCustomer(int id) {
    printf("Deleting customer...\n");
    
    // Build the query
    sprintf(g_query, "DELETE FROM Customers WHERE ID = %d", id);
    
    // Allocate a command
    g_retcode = ct_cmd_alloc(g_connection, &g_command);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to allocate command!\n");
        error = 27;
        return;
    }
    
    // Initialize the command
    g_retcode = ct_command(g_command, CS_LANG_CMD, g_query, 
                         strlen(g_query), CS_UNUSED);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to initialize command!\n");
        error = 28;
        return;
    }
    
    // Send the command
    g_retcode = ct_send(g_command);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to send command!\n");
        error = 29;
        return;
    }
    
    // Process all results
    CS_INT result_type;
    CS_INT rows_affected = 0;
    
    while ((g_retcode = ct_results(g_command, &result_type)) == CS_SUCCEED) {
        if (result_type == CS_CMD_SUCCEED) {
            // Get the number of rows affected
            g_retcode = ct_res_info(g_command, CS_ROW_COUNT, &rows_affected, 
                                  CS_UNUSED, NULL);
            if (g_retcode != CS_SUCCEED) {
                printf("Failed to get rows affected!\n");
                error = 30;
                break;
            }
        }
    }
    
    if (g_retcode != CS_END_RESULTS) {
        printf("Error in results processing!\n");
        error = 31;
        return;
    }
    
    // Drop the command
    ct_cmd_drop(g_command);
    
    if (rows_affected > 0) {
        printf("Customer deleted successfully! Rows affected: %d\n", rows_affected);
    } else {
        printf("No customer found with ID %d\n", id);
    }
}

// Function to disconnect from the database
void DisconnectFromDatabase() {
    printf("Disconnecting from database...\n");
    
    // Close the connection
    g_retcode = ct_close(g_connection, CS_UNUSED);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to close connection!\n");
        error = 32;
        return;
    }
    
    // Drop the connection
    g_retcode = ct_con_drop(g_connection);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to drop connection!\n");
        error = 33;
        return;
    }
    
    // Exit Client-Library
    g_retcode = ct_exit(g_context, CS_UNUSED);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to exit Client-Library!\n");
        error = 34;
        return;
    }
    
    // Drop the context
    g_retcode = cs_ctx_drop(g_context);
    if (g_retcode != CS_SUCCEED) {
        printf("Failed to drop context!\n");
        error = 35;
        return;
    }
    
    printf("Disconnected from database successfully!\n");
}

// Main function to run everything
int main() {
    // Display welcome message
    printf("*************************************\n");
    printf("* Sybase Customer Database Program  *\n");
    printf("* Version 1.0                       *\n");
    printf("* By: John Smith                    *\n");
    printf("*************************************\n\n");
    
    // Connect to database
    ConnectToDatabase();
    check_error();
    
    // Show menu and get user choice
    int choice;
    int customer_id;
    char customer_name[100];
    char customer_address[200];
    double customer_balance;
    Customer customer;
    
    do {
        printf("\nMenu:\n");
        printf("1. Add a customer\n");
        printf("2. Get customer details\n");
        printf("3. Update customer balance\n");
        printf("4. Delete a customer\n");
        printf("5. Exit\n");
        printf("Enter your choice (1-5): ");
        scanf("%d", &choice);
        
        switch (choice) {
            case 1:  // Add a customer
                printf("Enter customer ID: ");
                scanf("%d", &customer_id);
                printf("Enter customer name: ");
                scanf(" %[^\n]", customer_name);  // Read until newline
                printf("Enter customer address: ");
                scanf(" %[^\n]", customer_address);
                printf("Enter customer balance: ");
                scanf("%lf", &customer_balance);
                
                AddCustomer(customer_id, customer_name, customer_address, customer_balance);
                check_error();
                break;
            
            case 2:  // Get customer details
                printf("Enter customer ID: ");
                scanf("%d", &customer_id);
                
                customer = GetCustomer(customer_id);
                check_error();
                
                if (customer.id != 0) {
                    printf("\nCustomer Details:\n");
                    printf("ID: %d\n", customer.id);
                    printf("Name: %s\n", customer.name);
                    printf("Address: %s\n", customer.address);
                    printf("Balance: $%.2f\n", customer.balance);
                }
                break;
            
            case 3:  // Update customer balance
                printf("Enter customer ID: ");
                scanf("%d", &customer_id);
                printf("Enter new balance: ");
                scanf("%lf", &customer_balance);
                
                UpdateCustomerBalance(customer_id, customer_balance);
                check_error();
                break;
            
            case 4:  // Delete a customer
                printf("Enter customer ID: ");
                scanf("%d", &customer_id);
                
                DeleteCustomer(customer_id);
                check_error();
                break;
            
            case 5:  // Exit
                printf("Exiting program...\n");
                break;
            
            default:
                printf("Invalid choice! Please enter a number between 1 and 5.\n");
                break;
        }
    } while (choice != 5);
    
    // Disconnect from database
    DisconnectFromDatabase();
    check_error();
    
    printf("\nThank you for using this program!\n");
    
    return 0;
}