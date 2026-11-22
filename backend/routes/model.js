const express = require('express');
const fs = require('fs');
const path = require('path');
const pool = require('../db');
const { requireAuth } = require('../middleware/auth');

const router = express.Router();

/* Model performance metrics are kept offline in ml/evaluate.py and
   backend/model_metrics.json for Chapter 4 evaluation rather than
   being displayed on the website. */
router.get('/status', requireAuth('teacher'), async (req, res) => {
  try {
    const c = await pool.query(
      `select count(*) filter (where answers is not null) as with_answers
         from students`
    );
    const hasMetricsFile = fs.existsSync(path.join(__dirname, '..', 'model_metrics.json'));
    res.json({
      with_answers: Number(c.rows[0].with_answers),
      has_metrics_file: hasMetricsFile
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Could not load status.' });
  }
});

module.exports = router;