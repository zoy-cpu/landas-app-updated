const express = require('express');
const pool = require('../db');
const { requireAuth } = require('../middleware/auth');

const router = express.Router();

/* ---- student: save their own quiz result ---- */
router.put('/me/result', requireAuth('student'), async (req, res) => {
  const { style, confidence, answers } = req.body;
  if (!['V', 'A', 'R', 'K'].includes(style) || typeof confidence !== 'number') {
    return res.status(400).json({ error: 'Invalid result payload.' });
  }
  // answers is optional (older clients) but, when sent, must be exactly 16 letters V/A/R/K.
  let answersJson = null;
  if (answers !== undefined && answers !== null) {
    const ok = Array.isArray(answers) && answers.length === 16 &&
      answers.every(a => ['V', 'A', 'R', 'K'].includes(a));
    if (!ok) return res.status(400).json({ error: 'Invalid answers payload.' });
    answersJson = JSON.stringify(answers);
  }
  try {
    await pool.query(
      `update students
          set style=$1, confidence=$2, completed_at=now(),
              answers=coalesce($4::jsonb, answers)
        where id=$3`,
      [style, confidence, req.user.id, answersJson]
    );
    res.json({ ok: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Could not save your result.' });
  }
});

/* ---- teacher: roster for their own school + grade only ----
   The school/grade come from the teacher's own profile server-side
   (not from query params) so a teacher can never fetch another
   school's or grade's roster by editing the request. */
router.get('/roster', requireAuth('teacher'), async (req, res) => {
  try {
    const prof = await pool.query(`select school, grade from teacher_profiles where id=$1`, [req.user.id]);
    const { school, grade } = prof.rows[0];

    const roster = await pool.query(
      `select id, name, school, grade, section, style, confidence, completed_at
       from students
       where school=$1 and grade=$2 and style is not null
       order by completed_at desc`,
      [school, grade]
    );

    const now = Date.now();
    const withFlags = roster.rows.map(r => ({
      ...r,
      isNew: now - new Date(r.completed_at).getTime() < 10 * 60 * 1000
    }));
    res.json(withFlags);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Could not load roster.' });
  }
});

module.exports = router;