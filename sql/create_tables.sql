-- ==============================================================================
-- create_tables.sql
-- Schema definition for the Healthcare Operations Analytics Platform.
-- Defines the canonical 'patients' table and supporting indexes.
-- Standardized to snake_case column names to ensure 100% consistency with
-- the Python ETL pipeline (src/transform.py, src/load_sql.py) and SQL analytics.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. SQLite Schema (Default Local Setup)
-- ------------------------------------------------------------------------------
DROP TABLE IF EXISTS patients;

CREATE TABLE patients (
    patient_id           INTEGER PRIMARY KEY,
    name                 VARCHAR(150) NOT NULL,
    age                  INTEGER NOT NULL,
    gender               VARCHAR(20),
    blood_type           VARCHAR(10),
    medical_condition    VARCHAR(100),
    admission_date       DATE NOT NULL,
    doctor               VARCHAR(150) NOT NULL,
    hospital             VARCHAR(150) NOT NULL,
    insurance_provider   VARCHAR(100),
    billing_amount       DECIMAL(12,2) NOT NULL,
    room_number          INTEGER,
    admission_type       VARCHAR(50),
    discharge_date       DATE,
    medication           VARCHAR(100),
    test_results         VARCHAR(50),
    department           VARCHAR(100),
    length_of_stay       INTEGER,
    age_group            VARCHAR(50),
    admission_year       INTEGER,
    admission_month      INTEGER,
    admission_month_name VARCHAR(20)
);

CREATE INDEX ix_patients_department ON patients(department);
CREATE INDEX ix_patients_doctor ON patients(doctor);
CREATE INDEX ix_patients_admission_date ON patients(admission_date);
CREATE INDEX ix_patients_medical_condition ON patients(medical_condition);


-- ------------------------------------------------------------------------------
-- 2. Microsoft SQL Server Schema (Enterprise / Production Setup)
-- ------------------------------------------------------------------------------
/*
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'HospitalDB')
BEGIN
    CREATE DATABASE HospitalDB;
END;
GO

USE HospitalDB;
GO

IF OBJECT_ID('dbo.patients', 'U') IS NOT NULL
    DROP TABLE dbo.patients;
GO

CREATE TABLE dbo.patients (
    patient_id           INT PRIMARY KEY,
    name                 NVARCHAR(150) NOT NULL,
    age                  INT NOT NULL,
    gender               NVARCHAR(20),
    blood_type           NVARCHAR(10),
    medical_condition    NVARCHAR(100),
    admission_date       DATE NOT NULL,
    doctor               NVARCHAR(150) NOT NULL,
    hospital             NVARCHAR(150) NOT NULL,
    insurance_provider   NVARCHAR(100),
    billing_amount       DECIMAL(12,2) NOT NULL,
    room_number          INT,
    admission_type       NVARCHAR(50),
    discharge_date       DATE,
    medication           NVARCHAR(100),
    test_results         NVARCHAR(50),
    department           NVARCHAR(100),
    length_of_stay       INT,
    age_group            NVARCHAR(50),
    admission_year       INT,
    admission_month      INT,
    admission_month_name NVARCHAR(20)
);
GO

CREATE INDEX ix_patients_department ON dbo.patients(department);
CREATE INDEX ix_patients_doctor ON dbo.patients(doctor);
CREATE INDEX ix_patients_admission_date ON dbo.patients(admission_date);
CREATE INDEX ix_patients_medical_condition ON dbo.patients(medical_condition);
GO
*/
