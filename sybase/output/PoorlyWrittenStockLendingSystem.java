/**
 * @file PoorlyWrittenStockLendingSystem.java
 * 
 * JDBC code converted from Sybase to Microsoft SQL Server by Sybase Java JDBC Converter
 * Conversion date: 2025-03-01 13:14:35
 */

package com.atroxfortuna.stocklending;

import com.microsoft.sqlserver.jdbc.SQLServerDriver;
import com.microsoft.sqlserver.jdbc.SQLServerDataSource;
import java.sql.*;
import java.util.*;
import java.math.BigDecimal;

/**
 * Poorly Written Stock Lending System - Example of bad coding practices
 * 
 * This class demonstrates poor coding practices such as lack of error handling,
 * hardcoded values, and inefficient resource management.
 */
public class PoorlyWrittenStockLendingSystem {
    
    // Hardcoded database connection details
    private static final String DB_URL = "jdbc:sqlserver://localhost5000;mydatabase";
    private static final String DB_USER = "sa";
    private static final String DB_PASSWORD = "password";
    
    // Hardcoded SQL queries
    private static final String SQL_GET_SECURITIES = "SELECT * FROM securities";
    private static final String SQL_INSERT_LOAN = "INSERT INTO stock_loans (loan_ref, counterparty_id, security_id, quantity) VALUES (?, ?, ?, ?)";
    private static final String SQL_UPDATE_LOAN = "UPDATE stock_loans SET quantity = ? WHERE loan_ref = ?";
    private static final String SQL_DELETE_LOAN = "DELETE FROM stock_loans WHERE loan_ref = ?";
    
    public static void main(String[] args) {
        PoorlyWrittenStockLendingSystem system = new PoorlyWrittenStockLendingSystem();
        system.getSecurities();
        system.recordLoan("L123", 1, 1, new BigDecimal("1000"));
        system.updateLoan("L123", new BigDecimal("1500"));
        system.deleteLoan("L123");
        system.getTotalLoans();
        system.getLoanDetails("L123");
    }
    
    public void getSecurities() {
        Connection conn = null;
        Statement stmt = null;
        ResultSet rs = null;
        
        try {
            // Establish connection
            conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
            stmt = conn.createStatement();
            rs = stmt.executeQuery(SQL_GET_SECURITIES);
            
            // Process result set
            while (rs.next()) {
                System.out.println("Security ID: " + rs.getInt("security_id"));
                System.out.println("Ticker: " + rs.getString("ticker"));
                System.out.println("Description: " + rs.getString("description"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            // Close resources without proper error handling
            try { if (rs != null) rs.close(); } catch (SQLException e) { e.printStackTrace(); }
            try { if (stmt != null) stmt.close(); } catch (SQLException e) { e.printStackTrace(); }
            try { if (conn != null) conn.close(); } catch (SQLException e) { e.printStackTrace(); }
        }
    }
    
    public void recordLoan(String loanRef, int counterpartyId, int securityId, BigDecimal quantity) {
        Connection conn = null;
        PreparedStatement pstmt = null;
        
        try {
            // Establish connection
            conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
            pstmt = conn.prepareStatement(SQL_INSERT_LOAN);
            
            // Set parameters
            pstmt.setString(1, loanRef);
            pstmt.setInt(2, counterpartyId);
            pstmt.setInt(3, securityId);
            pstmt.setBigDecimal(4, quantity);
            
            // Execute update
            pstmt.executeUpdate();
            System.out.println("Loan recorded successfully");
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            // Close resources without proper error handling
            try { if (pstmt != null) pstmt.close(); } catch (SQLException e) { e.printStackTrace(); }
            try { if (conn != null) conn.close(); } catch (SQLException e) { e.printStackTrace(); }
        }
    }
    
    public void updateLoan(String loanRef, BigDecimal newQuantity) {
        Connection conn = null;
        PreparedStatement pstmt = null;
        
        try {
            // Establish connection
            conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
            pstmt = conn.prepareStatement(SQL_UPDATE_LOAN);
            
            // Set parameters
            pstmt.setBigDecimal(1, newQuantity);
            pstmt.setString(2, loanRef);
            
            // Execute update
            pstmt.executeUpdate();
            System.out.println("Loan updated successfully");
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            // Close resources without proper error handling
            try { if (pstmt != null) pstmt.close(); } catch (SQLException e) { e.printStackTrace(); }
            try { if (conn != null) conn.close(); } catch (SQLException e) { e.printStackTrace(); }
        }
    }
    
    public void deleteLoan(String loanRef) {
        Connection conn = null;
        PreparedStatement pstmt = null;
        
        try {
            // Establish connection
            conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
            pstmt = conn.prepareStatement(SQL_DELETE_LOAN);
            
            // Set parameters
            pstmt.setString(1, loanRef);
            
            // Execute update
            pstmt.executeUpdate();
            System.out.println("Loan deleted successfully");
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            // Close resources without proper error handling
            try { if (pstmt != null) pstmt.close(); } catch (SQLException e) { e.printStackTrace(); }
            try { if (conn != null) conn.close(); } catch (SQLException e) { e.printStackTrace(); }
        }
    }
    
    public void getTotalLoans() {
        Connection conn = null;
        Statement stmt = null;
        ResultSet rs = null;
        
        try {
            // Establish connection
            conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
            stmt = conn.createStatement();
            rs = stmt.executeQuery("SELECT COUNT(*) AS total_loans FROM stock_loans");
            
            // Process result set
            if (rs.next()) {
                int totalLoans = rs.getInt("total_loans");
                System.out.println("Total Loans: " + totalLoans);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            // Close resources without proper error handling
            try { if (rs != null) rs.close(); } catch (SQLException e) { e.printStackTrace(); }
            try { if (stmt != null) stmt.close(); } catch (SQLException e) { e.printStackTrace(); }
            try { if (conn != null) conn.close(); } catch (SQLException e) { e.printStackTrace(); }
        }
    }
    
    public void getLoanDetails(String loanRef) {
        Connection conn = null;
        PreparedStatement pstmt = null;
        ResultSet rs = null;
        
        try {
            // Establish connection
            conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
            pstmt = conn.prepareStatement("SELECT * FROM stock_loans WHERE loan_ref = ?");
            pstmt.setString(1, loanRef);
            rs = pstmt.executeQuery();
            
            // Process result set
            if (rs.next()) {
                System.out.println("Loan Ref: " + rs.getString("loan_ref"));
                System.out.println("Counterparty ID: " + rs.getInt("counterparty_id"));
                System.out.println("Security ID: " + rs.getInt("security_id"));
                System.out.println("Quantity: " + rs.getBigDecimal("quantity"));
                System.out.println("Loan Rate: " + rs.getBigDecimal("loan_rate"));
                System.out.println("Start Date: " + rs.getDate("start_date"));
                System.out.println("End Date: " + rs.getDate("end_date"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            // Close resources without proper error handling
            try { if (rs != null) rs.close(); } catch (SQLException e) { e.printStackTrace(); }
            try { if (pstmt != null) pstmt.close(); } catch (SQLException e) { e.printStackTrace(); }
            try { if (conn != null) conn.close(); } catch (SQLException e) { e.printStackTrace(); }
        }
    }
}
