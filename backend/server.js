require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const rateLimit = require('express-rate-limit');

const authRoutes = require('./routes/auth');
const studentRoutes = require('./routes/students');
const modelRoutes = require('./routes/model');

const app = express();
app.set('trust proxy', 1);

// --- CORS ---
// Restrict to your real frontend origin(s) instead of allowing any site.
// Set ALLOWED_ORIGINS in your .env as a comma-separated list, e.g.:
//   ALLOWED_ORIGINS=https://your-frontend-domain.com,http://localhost:5173
const allowedOrigins = (process.env.ALLOWED_ORIGINS || '')
  .split(',')
  .map(o => o.trim())
  .filter(Boolean);

app.use(cors({
  origin: function (origin, callback) {
    if (!origin || allowedOrigins.length === 0 || allowedOrigins.includes(origin)) {
      return callback(null, true);
    }
    return callback(new Error('Not allowed by CORS'));
  }
}));

app.use(express.json());

app.use(express.static(path.join(__dirname, 'www')));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'www', 'index.html'));
});

app.get('/api/health', (req, res) => res.json({ ok: true }));

// --- Rate limiting on auth routes ---
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 20,                  // 20 requests per window per IP
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many attempts. Please try again later.' }
});

app.use('/api/auth', authLimiter, authRoutes);
app.use('/api/students', studentRoutes);
app.use('/api/model', modelRoutes);

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`Landas backend listening on port ${PORT}`));