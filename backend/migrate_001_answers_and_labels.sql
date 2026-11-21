-- Run once against your existing database (Supabase SQL editor or psql).
-- Safe to run more than once.
alter table students add column if not exists answers jsonb;          -- the 16 answers, e.g. ["V","K","A",...]
alter table students add column if not exists teacher_label text
  check (teacher_label in ('V','A','R','K'));                         -- learning style confirmed by the teacher
alter table students add column if not exists labeled_at timestamptz;
