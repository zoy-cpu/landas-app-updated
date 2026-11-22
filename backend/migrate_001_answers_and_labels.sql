-- Run once against your existing database (Supabase SQL editor or psql).
-- Safe to run more than once.
alter table students add column if not exists answers jsonb; -- the 16 answers, e.g. ["V","K","A",...]

-- Optional: uncomment and run only after backing up your database if you want
-- to remove the teacher labeling columns from PostgreSQL:
-- alter table students drop column if exists teacher_label;
-- alter table students drop column if exists labeled_at;