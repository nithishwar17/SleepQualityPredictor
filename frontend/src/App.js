// frontend/src/App.js
import React from 'react';
import './App.css';

function App() {
  const [form, setForm] = React.useState({
    age: '35', gender: 'Female', sleep_duration: '7.5', bedtime: '23:00', wake_time: '06:30',
    awakenings: '1', caffeine_consumption: '50', alcohol_consumption: '0', smoking_status: 'No', exercise_frequency: '3',
  });
  const [result, setResult] = React.useState(null);
  const [loading, setLoading] = React.useState(false);

  // ... (handleChange and handleSubmit functions remain the same)
  function handleChange(e) {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Sleep Quality Predictor 💤</h1>
        <p>Enter your daily sleep metrics to predict your sleep quality.</p>
      </header>
      
      <main className="app-main">
        <div className="card form-card">
          {/* ... (The form JSX remains the same) */}
          <h2>Your Sleep Metrics</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <label> Age <input name="age" type="number" value={form.age} onChange={handleChange} required /> </label>
              <label> Gender <select name="gender" value={form.gender} onChange={handleChange}><option>Female</option><option>Male</option></select> </label>
              <label> Sleep duration (hours) <input name="sleep_duration" type="number" step="0.5" value={form.sleep_duration} onChange={handleChange} required /> </label>
              <label> Bedtime <input name="bedtime" type="time" value={form.bedtime} onChange={handleChange} required /> </label>
              <label> Wake-up time <input name="wake_time" type="time" value={form.wake_time} onChange={handleChange} required /> </label>
              <label> Awakenings (count) <input name="awakenings" type="number" min="0" value={form.awakenings} onChange={handleChange} required /> </label>
              <label> Caffeine (mg) <input name="caffeine_consumption" type="number" min="0" value={form.caffeine_consumption} onChange={handleChange} required /> </label>
              <label> Alcohol (units/week) <input name="alcohol_consumption" type="number" min="0" value={form.alcohol_consumption} onChange={handleChange} required /> </label>
              <label> Smoking status <select name="smoking_status" value={form.smoking_status} onChange={handleChange}><option>No</option><option>Yes</option></select> </label>
              <label> Exercise (days/week) <input name="exercise_frequency" type="number" min="0" max="7" value={form.exercise_frequency} onChange={handleChange} required /> </label>
            </div>
            <button type="submit" disabled={loading}> {loading ? 'Analyzing...' : 'Predict My Sleep Quality'} </button>
          </form>
        </div>

        {result && (
          <div className={`card result-card result-${result.prediction_label?.toLowerCase()}`}>
            {result.error ? ( <div className="error-message">Error: {result.error}</div> ) : (
              <>
                <h3>Prediction Result</h3>
                <div className="prediction-display">
                  Your sleep quality is likely to be:
                  <span className="prediction-label">{result.prediction_label}</span>
                </div>
                <div className="probabilities">
                  <span>Poor: {Math.round(result.probabilities.Poor * 100)}%</span>
                  <span>Average: {Math.round(result.probabilities.Average * 100)}%</span>
                  <span>Good: {Math.round(result.probabilities.Good * 100)}%</span>
                </div>
                
                {/* --- ADDING THE TIPS SECTION --- */}
                <div className="tips-section">
                  <h4>💡 Personalized Tips:</h4>
                  <ul>
                    {result.tips?.map((tip, index) => (
                      <li key={index}>{tip}</li>
                    ))}
                  </ul>
                </div>
              </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
