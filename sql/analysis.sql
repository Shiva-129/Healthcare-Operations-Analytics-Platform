-- ==============================================================================
-- analysis.sql
-- Business analysis queries for the Healthcare Operations Analytics Platform.
-- Written against the standardized snake_case schema of the 'patients' table.
-- Compatible with SQLite (default local) and Microsoft SQL Server (production).
-- ==============================================================================

-- 1. Total Patients
SELECT COUNT(*) AS total_patients
FROM patients;

-- 2. Monthly Admissions Trend
SELECT 
    admission_year, 
    admission_month, 
    admission_month_name, 
    COUNT(*) AS admissions
FROM patients
GROUP BY admission_year, admission_month, admission_month_name
ORDER BY admission_year, admission_month;

-- 3. Department-wise Revenue & Patient Volume
SELECT 
    department, 
    ROUND(SUM(billing_amount), 2) AS total_revenue, 
    COUNT(*) AS patient_count,
    ROUND(AVG(billing_amount), 2) AS avg_billing_per_patient
FROM patients
GROUP BY department
ORDER BY total_revenue DESC;

-- 4. Top 10 Doctors by Patient Count
-- For SQL Server: replace 'LIMIT 10' with 'SELECT TOP 10 ...'
SELECT 
    doctor, 
    COUNT(*) AS patients_handled, 
    ROUND(SUM(billing_amount), 2) AS revenue_generated
FROM patients
GROUP BY doctor
ORDER BY patients_handled DESC
LIMIT 10;

-- 5. Average Billing Amount by Department
SELECT 
    department, 
    ROUND(AVG(billing_amount), 2) AS avg_billing
FROM patients
GROUP BY department
ORDER BY avg_billing DESC;

-- 6. Average Length of Stay by Department (in days)
SELECT 
    department, 
    ROUND(AVG(CAST(length_of_stay AS FLOAT)), 2) AS avg_length_of_stay
FROM patients
GROUP BY department
ORDER BY avg_length_of_stay DESC;

-- 7. Insurance Provider Analysis
SELECT 
    insurance_provider, 
    COUNT(*) AS patient_count, 
    ROUND(SUM(billing_amount), 2) AS total_billed,
    ROUND(AVG(billing_amount), 2) AS avg_billed
FROM patients
GROUP BY insurance_provider
ORDER BY patient_count DESC;

-- 8. Gender Distribution & Percentage of Total
SELECT 
    gender, 
    COUNT(*) AS patient_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM patients), 2) AS percentage
FROM patients
GROUP BY gender;

-- 9. Age Group Distribution
SELECT 
    age_group, 
    COUNT(*) AS patient_count
FROM patients
GROUP BY age_group
ORDER BY patient_count DESC;

-- 10. Yearly Patient Trends & Revenue
SELECT 
    admission_year, 
    COUNT(*) AS total_patients, 
    ROUND(SUM(billing_amount), 2) AS total_revenue
FROM patients
GROUP BY admission_year
ORDER BY admission_year;

-- 11. Revenue Ranking by Department (Window Function: RANK)
SELECT 
    department,
    ROUND(SUM(billing_amount), 2) AS total_revenue,
    RANK() OVER (ORDER BY SUM(billing_amount) DESC) AS revenue_rank
FROM patients
GROUP BY department;

-- 12. Doctor Ranking within each Department (Window Function: ROW_NUMBER)
SELECT 
    department, 
    doctor, 
    patients_handled,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY patients_handled DESC) AS rank_in_dept
FROM (
    SELECT department, doctor, COUNT(*) AS patients_handled
    FROM patients
    GROUP BY department, doctor
) AS dept_doctor_counts;

-- 13. CTE: High-Cost Patients (Above Overall Average Billing)
WITH avg_billing AS (
    SELECT AVG(billing_amount) AS overall_avg FROM patients
)
SELECT 
    p.patient_id, 
    p.name, 
    p.department, 
    p.billing_amount
FROM patients p
CROSS JOIN avg_billing a
WHERE p.billing_amount > a.overall_avg
ORDER BY p.billing_amount DESC
LIMIT 50;

-- 14. CTE: Month-over-Month Revenue Growth (Window Function: LAG)
WITH monthly_revenue AS (
    SELECT 
        admission_year, 
        admission_month, 
        ROUND(SUM(billing_amount), 2) AS revenue
    FROM patients
    GROUP BY admission_year, admission_month
)
SELECT 
    admission_year, 
    admission_month, 
    revenue,
    LAG(revenue) OVER (ORDER BY admission_year, admission_month) AS prev_month_revenue,
    ROUND(revenue - LAG(revenue) OVER (ORDER BY admission_year, admission_month), 2) AS revenue_change
FROM monthly_revenue
ORDER BY admission_year, admission_month;

-- 15. Most Frequent Medical Conditions
SELECT 
    medical_condition, 
    COUNT(*) AS cases,
    ROUND(AVG(length_of_stay), 2) AS avg_stay_days
FROM patients
GROUP BY medical_condition
ORDER BY cases DESC;

-- 16. Test Results Breakdown
SELECT 
    test_results, 
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM patients), 2) AS percentage
FROM patients
GROUP BY test_results;

-- 17. Reusable View: Executive Summary (for BI & Dashboards)
DROP VIEW IF EXISTS vw_executive_summary;
CREATE VIEW vw_executive_summary AS
SELECT
    COUNT(*)                                  AS total_patients,
    ROUND(SUM(billing_amount), 2)             AS total_revenue,
    ROUND(AVG(billing_amount), 2)             AS avg_billing,
    ROUND(AVG(CAST(length_of_stay AS FLOAT)), 2) AS avg_length_of_stay
FROM patients;

-- 18. Reusable View: Department Performance Summary
DROP VIEW IF EXISTS vw_department_performance;
CREATE VIEW vw_department_performance AS
SELECT
    department,
    COUNT(*)                                   AS patient_count,
    ROUND(SUM(billing_amount), 2)              AS total_revenue,
    ROUND(AVG(billing_amount), 2)              AS avg_billing,
    ROUND(AVG(CAST(length_of_stay AS FLOAT)), 2) AS avg_length_of_stay
FROM patients
GROUP BY department;
