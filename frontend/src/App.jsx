import React, { useState } from 'react';
import { 
  Sparkles, 
  Cpu, 
  TrendingUp, 
  Trophy, 
  Film, 
  Landmark, 
  Send, 
  RotateCcw, 
  CheckCircle2, 
  AlertCircle,
  Newspaper,
  Loader2
} from 'lucide-react';

const SAMPLES = {
  tech: "Quantum computing breakthrough allows processors to simulate complex chemical reactions in seconds, opening new frontiers for pharmaceutical research and material science.",
  business: "Global central banks hint at interest rate cuts as inflation figures drop below target levels, sparking a surge in global stock indices and tech equities.",
  sport: "Underdog team secures a sensational last-minute victory in the championship final with a stunning 30-yard volley in injury time.",
  entertainment: "Blockbuster film breaks opening weekend box office records worldwide, receiving overwhelming critical acclaim for visual effects and musical score.",
  politics: "Parliament passes landmark climate legislation aiming for net-zero carbon emissions, introducing stricter regulations for industrial energy producers."
};

const CATEGORY_META = {
  tech: { icon: Cpu, label: "Tech", colorClass: "tech" },
  business: { icon: TrendingUp, label: "Business", colorClass: "business" },
  sport: { icon: Trophy, label: "Sport", colorClass: "sport" },
  entertainment: { icon: Film, label: "Entertainment", colorClass: "entertainment" },
  politics: { icon: Landmark, label: "Politics", colorClass: "politics" }
};

export default function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    if (!text.trim()) {
      setError("Please enter or select a news article text.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Unable to connect to prediction backend.');
    } finally {
      setLoading(false);
    }
  };

  const loadSample = (catKey) => {
    setText(SAMPLES[catKey]);
    setError(null);
  };

  const clearAll = () => {
    setText('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="container">
      <header className="header">
        <div className="title-badge">
          <Sparkles size={14} /> DistilBERT + PEFT (LoRA) Powered
        </div>
        <h1 className="main-title">BBC News Text Classifier</h1>
        <p className="subtitle">
          Instantly classify news articles into 5 categories using fine-tuned transformer language models.
        </p>
      </header>

      <main className="glass-card">
        {/* Sample Selection */}
        <div className="sample-section">
          <div className="section-label">
            <Newspaper size={16} /> Quick Try Sample Articles:
          </div>
          <div className="sample-pills">
            {Object.keys(SAMPLES).map((cat) => {
              const MetaIcon = CATEGORY_META[cat].icon;
              return (
                <button
                  key={cat}
                  className={`pill-btn ${cat}`}
                  onClick={() => loadSample(cat)}
                >
                  <MetaIcon size={14} />
                  {CATEGORY_META[cat].label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Input Area */}
        <div className="textarea-container">
          <textarea
            className="custom-textarea"
            placeholder="Paste or type a news article paragraph here to predict its topic..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="textarea-footer">
            <span>{text.length} characters</span>
            {text && (
              <button className="clear-btn" onClick={clearAll}>
                Clear text
              </button>
            )}
          </div>
        </div>

        {/* Action Button */}
        <div className="action-bar">
          <button
            className="predict-btn"
            onClick={handlePredict}
            disabled={loading || !text.trim()}
          >
            {loading ? (
              <>
                <Loader2 size={18} className="spinner" /> Predicting...
              </>
            ) : (
              <>
                <Send size={18} /> Classify Article
              </>
            )}
          </button>
        </div>

        {/* Error Notification */}
        {error && (
          <div className="error-banner">
            <AlertCircle size={18} /> {error}
          </div>
        )}

        {/* Result Output */}
        {result && (
          <div className="result-card">
            <div className="result-header">
              <div className="predicted-badge-group">
                <span className="section-label" style={{ margin: 0 }}>Predicted Category:</span>
                {(() => {
                  const cat = result.predicted_category;
                  const MetaIcon = CATEGORY_META[cat]?.icon || Newspaper;
                  const colorClass = CATEGORY_META[cat]?.colorClass || 'tech';
                  return (
                    <div className={`category-tag ${colorClass}`}>
                      <MetaIcon size={20} /> {cat}
                    </div>
                  );
                })()}
              </div>
              <div className="confidence-val">
                Confidence: <span>{(result.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>

            <div className="probabilities-title">Model Confidence Distribution</div>
            <div className="prob-list">
              {Object.entries(result.probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([catName, probVal]) => {
                  const pct = (probVal * 100).toFixed(1);
                  const colorClass = CATEGORY_META[catName]?.colorClass || 'tech';
                  return (
                    <div className="prob-item" key={catName}>
                      <span className="prob-name">{catName}</span>
                      <div className="prob-bar-container">
                        <div
                          className={`prob-bar-fill ${colorClass}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="prob-percent">{pct}%</span>
                    </div>
                  );
                })}
            </div>
          </div>
        )}
      </main>

      <footer>
        Fine-tuned Model: DistilBERT-base-uncased with LoRA Adapter | React + FastAPI Architecture
      </footer>
    </div>
  );
}
