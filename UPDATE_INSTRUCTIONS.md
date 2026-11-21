# What changed and how to deploy it

## Changes
- `backend/migrate_001_answers_and_labels.sql` – NEW. Adds `answers`, `teacher_label`, `labeled_at` to `students`.
- `backend/routes/students.js` – saves the 16 answers; new `PUT /api/students/:id/label` (teacher confirms/corrects a style); roster now returns `id` and `teacher_label`.
- `backend/routes/model.js` + `server.js` – NEW `GET /api/model/metrics` (teacher only).
- `backend/www/index.html` and `www/index.html` – quiz sends answers; teacher student page has "Does this match what you observe?"; new "Model performance" page.
- `ml/evaluate.py` – NEW. Trains Decision Tree + Neural Network and compares them with the current scoring rule.

## Deploy
1. Run `backend/migrate_001_answers_and_labels.sql` once on your database (Supabase SQL editor).
2. Commit and push to GitHub; Render redeploys automatically.
3. Students must retake the quiz (old results have no saved answers). Teachers confirm each student's style.
4. When you have 100+ confirmed students, on your computer:
   cd ml
   pip install pandas scikit-learn matplotlib psycopg2-binary python-dotenv
   python evaluate.py
   Then commit `backend/model_metrics.json` and push. Tables and the confusion-matrix image for your paper are in `ml/output/`.
