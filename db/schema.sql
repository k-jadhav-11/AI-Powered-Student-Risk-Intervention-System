-- Relational schema for the Student Performance Analytics CLI 

CREATE TABLE IF NOT EXISTS students (
    student_id   INT AUTO_INCREMENT PRIMARY KEY,
    name         VARCHAR(150) NOT NULL,
    roll_number  VARCHAR(50) UNIQUE NOT NULL,
    class_name   VARCHAR(20)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS subjects (
    subject_id   INT AUTO_INCREMENT PRIMARY KEY,
    subject_name VARCHAR(100) UNIQUE NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS marks (
    mark_id      INT AUTO_INCREMENT PRIMARY KEY,
    student_id   INT NOT NULL,
    subject_id   INT NOT NULL,
    test_name    VARCHAR(50) NOT NULL,
    score        DECIMAL(5,2) NOT NULL,
    max_score    DECIMAL(5,2) NOT NULL DEFAULT 100,
    test_date    DATE,
    CONSTRAINT fk_marks_student FOREIGN KEY (student_id) REFERENCES students(student_id),
    CONSTRAINT fk_marks_subject FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
    CONSTRAINT uq_student_subject_test UNIQUE (student_id, subject_id, test_name)
) ENGINE=InnoDB;

CREATE INDEX idx_marks_student ON marks(student_id);
CREATE INDEX idx_marks_subject ON marks(subject_id);
