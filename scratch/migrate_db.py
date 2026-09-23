import sqlite3
import shutil
import os

db_path = os.path.abspath('instance/smarteval.db')
bak_path = os.path.abspath('instance/smarteval.db.bak')

print(f"Target DB: {db_path}")

# Step 1: Backup
shutil.copy2(db_path, bak_path)
print(f"Backup created at: {bak_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get row count before
count_before = cursor.execute("SELECT count(*) FROM answer_evaluation").fetchone()[0]
rows_before = cursor.execute("SELECT evaluation_id, submission_id, question_id, ai_score, evaluation_status FROM answer_evaluation ORDER BY evaluation_id").fetchall()
print(f"Row count before migration: {count_before}")

# Disable foreign keys for schema migration
cursor.execute("PRAGMA foreign_keys = OFF")

try:
    cursor.execute("BEGIN TRANSACTION")

    # Create new table with ai_score nullable
    cursor.execute("""
    CREATE TABLE answer_evaluation_new (
        evaluation_id INTEGER NOT NULL, 
        submission_id INTEGER NOT NULL, 
        question_id INTEGER NOT NULL, 
        answer_id INTEGER, 
        extracted_answer TEXT NOT NULL, 
        ai_score FLOAT, 
        maximum_marks FLOAT NOT NULL, 
        criteria_scores JSON, 
        strengths JSON, 
        missing_elements JSON, 
        feedback TEXT, 
        ai_confidence FLOAT NOT NULL, 
        semantic_similarity_score FLOAT, 
        evaluation_status VARCHAR(50) NOT NULL, 
        needs_faculty_review BOOLEAN NOT NULL, 
        faculty_score FLOAT, 
        faculty_feedback TEXT, 
        error_message TEXT, 
        created_at DATETIME NOT NULL, 
        updated_at DATETIME NOT NULL, 
        PRIMARY KEY (evaluation_id), 
        FOREIGN KEY(submission_id) REFERENCES submission (submission_id) ON DELETE CASCADE, 
        FOREIGN KEY(question_id) REFERENCES question (question_id) ON DELETE CASCADE, 
        FOREIGN KEY(answer_id) REFERENCES submission_answer (answer_id) ON DELETE SET NULL
    )
    """)

    # Copy data exactly
    cursor.execute("""
    INSERT INTO answer_evaluation_new (
        evaluation_id, submission_id, question_id, answer_id, extracted_answer,
        ai_score, maximum_marks, criteria_scores, strengths, missing_elements,
        feedback, ai_confidence, semantic_similarity_score, evaluation_status,
        needs_faculty_review, faculty_score, faculty_feedback, error_message,
        created_at, updated_at
    )
    SELECT 
        evaluation_id, submission_id, question_id, answer_id, extracted_answer,
        ai_score, maximum_marks, criteria_scores, strengths, missing_elements,
        feedback, ai_confidence, semantic_similarity_score, evaluation_status,
        needs_faculty_review, faculty_score, faculty_feedback, error_message,
        created_at, updated_at
    FROM answer_evaluation
    """)

    # Drop old table
    cursor.execute("DROP TABLE answer_evaluation")

    # Rename new table
    cursor.execute("ALTER TABLE answer_evaluation_new RENAME TO answer_evaluation")

    # Recreate indexes
    cursor.execute("CREATE INDEX ix_answer_evaluation_submission_id ON answer_evaluation (submission_id)")
    cursor.execute("CREATE INDEX ix_answer_evaluation_question_id ON answer_evaluation (question_id)")
    cursor.execute("CREATE INDEX ix_answer_evaluation_evaluation_status ON answer_evaluation (evaluation_status)")
    cursor.execute("CREATE INDEX ix_answer_evaluation_answer_id ON answer_evaluation (answer_id)")

    conn.commit()
    print("Migration transaction committed successfully.")

except Exception as e:
    conn.rollback()
    print(f"Migration failed and was rolled back: {e}")
    raise
finally:
    cursor.execute("PRAGMA foreign_keys = ON")

# Verify foreign keys
fk_errors = cursor.execute("PRAGMA foreign_key_check").fetchall()
if fk_errors:
    print(f"WARNING: Foreign key violations detected: {fk_errors}")
else:
    print("Foreign key integrity check: PASSED (0 violations).")

# Row count after
count_after = cursor.execute("SELECT count(*) FROM answer_evaluation").fetchone()[0]
rows_after = cursor.execute("SELECT evaluation_id, submission_id, question_id, ai_score, evaluation_status FROM answer_evaluation ORDER BY evaluation_id").fetchall()
print(f"Row count after migration: {count_after}")

# Check table info
print("\n--- New PRAGMA table_info ---")
ai_score_col = None
for col in cursor.execute("PRAGMA table_info(answer_evaluation)"):
    print(col)
    if col[1] == 'ai_score':
        ai_score_col = col

print(f"\nai_score column info: {ai_score_col} (notnull={ai_score_col[3]})")

assert count_before == count_after, f"Row count mismatch: {count_before} vs {count_after}"
assert rows_before == rows_after, "Data rows mismatch!"
assert ai_score_col[3] == 0, "ai_score is still NOT NULL!"

print("\nAll migration assertions PASSED!")
conn.close()
