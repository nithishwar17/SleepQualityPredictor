// frontend/src/App.js
import React, { useState } from 'react';
import './App.css'; // Let's add some basic styling

function App() {
  const [form, setForm] = useState({
    age: '35',
    gender: 'Female',
    sleep_duration: '7',
    bedtime: '23:00',
    wake_time: '06:00',
    awakenings: '0',
    caffeine_consumption: '25',
    alcohol_consumption: '0',
    smoking_status: 'No',
    exercise_frequency: '3',
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

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
    <div className="container">
      <h1>Sleep Quality Predictor 💤</h1>
      <form onSubmit={handleSubmit}>
        <div className="grid">
          <label> Age <input name="age" type="number" value={form.age} onChange={handleChange} /> </label>
          <label> Gender
            <select name="gender" value={form.gender} onChange={handleChange}>
              <option>Female</option> <option>Male</option>
            </select>
          </label>
          <label> Sleep duration (hours) <input name="sleep_duration" type="number" step="0.5" value={form.sleep_duration} onChange={handleChange} /> </label>
          <label> Bedtime (HH:MM) <input name="bedtime" type="time" value={form.bedtime} onChange={handleChange} /> </label>
          <label> Wakeup time (HH:MM) <input name="wake_time" type="time" value={form.wake_time} onChange={handleChange} /> </label>
          <label> Awakenings (times per night) <input name="awakenings" type="number" value={form.awakenings} onChange={handleChange} /> </label>
          <label> Caffeine (mg) <input name="caffeine_consumption" type="number" value={form.caffeine_consumption} onChange={handleChange} /> </label>
          <label> Alcohol consumption (units per week) <input name="alcohol_consumption" type="number" value={form.alcohol_consumption} onChange={handleChange} /> </label>
          <label> Smoking status
            <select name="smoking_status" value={form.smoking_status} onChange={handleChange}>
              <option>No</option> <option>Yes</option>
            </select>
          </label>
          <label> Exercise frequency (days/week) <input name="exercise_frequency" type="number" value={form.exercise_frequency} onChange={handleChange} /> </label>
        </div>
        <button type="submit" disabled={loading}> {loading ? 'Analyzing...' : 'Predict My Sleep Quality'} </button>
      </form>

      {result && (
        <div className="result-card">
          {result.error ? ( <div className="error">Error: {result.error}</div> ) : (
            <>
              <h3>Prediction: <span className={`label-${result.prediction_label}`}>{result.prediction_label}</span></h3>
              <div className="probabilities">
                <p><strong>Confidence Score:</strong></p>
                <ul>
                  <li>Poor: {Math.round(result.probabilities.Poor * 100)}%</li>
                  <li>Average: {Math.round(result.probabilities.Average * 100)}%</li>
                  <li>Good: {Math.round(result.probabilities.Good * 100)}%</li>
                </ul>
              </div>
              <div className="tips">
                <strong>💡 Personalized Tips:</strong>
                <ul> {result.tips?.map((t, i) => <li key={i}>{t}</li>)} </ul>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;