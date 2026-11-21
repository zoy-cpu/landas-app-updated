const express = require('express');
const fs = require('fs');
const path = require('path');
const pool = require('../db');
const { requireAuth } = require('../middleware/auth');

const router = express.Router();

/* Teachers see how much evaluation data exists, plus the latest results
   written by ml/evaluate.py (backend/model_metrics.json) if it has been run. */
router.get('/metrics', requireAuth('teacher'), async (req, res) => {
  try {
    const c = await pool.query(
      `select count(*) filter (where answers is not null)                         as with_answers,
              count(*) filter (where answers is not null and teacher_label is not null) as labeled
         from students`
    );
    let metrics = null;
    try {
      metrics = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'model_metrics.json'), 'utf8'));
    } catch (e) { /* not generated yet */ }
    res.json({
      with_answers: Number(c.rows[0].with_answers),
      labeled: Number(c.rows[0].labeled),
      metrics
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Could not load model status.' });
  }
});

module.exports = router;
