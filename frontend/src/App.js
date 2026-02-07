import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const uploadImage = async () => {
    if (!file) {
      alert("Please select an MRI image first!");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("https://alzheimer-backend-xyz.onrender.com/predict", {


        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);
    } catch (error) {
      alert("Error connecting to backend. Is FastAPI running?");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-100 to-blue-200 flex items-center justify-center">
      <div className="bg-white shadow-2xl rounded-2xl p-8 w-[450px] text-center">

        <h1 className="text-2xl font-bold text-gray-800">
          🧠 Alzheimer Early Detection System
        </h1>

        <p className="text-gray-600 mt-2">
          Upload an MRI scan to analyze dementia stage
        </p>

        <div className="mt-6">
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files[0])}
            className="block w-full text-sm text-gray-500
                       file:mr-4 file:py-2 file:px-4
                       file:rounded-full file:border-0
                       file:text-sm file:font-semibold
                       file:bg-teal-50 file:text-teal-700
                       hover:file:bg-teal-100"
          />
        </div>

        <button
          onClick={uploadImage}
          className="mt-6 bg-teal-600 text-white px-6 py-2 rounded-lg
                     hover:bg-teal-700 transition"
        >
          Analyze MRI
        </button>

        {loading && (
          <p className="mt-4 text-gray-700 font-semibold">
            Analyzing... please wait ⏳
          </p>
        )}

        {result && (
          <div className="mt-6 p-4 border rounded-lg bg-teal-50">
            <h2 className="text-lg font-bold text-teal-800">Result</h2>
            <p className="mt-2">
              <span className="font-semibold">Predicted Class:</span>{" "}
              {result.predicted_class}
            </p>
            <p>
              <span className="font-semibold">Confidence:</span>{" "}
              {result.confidence_percentage}%
            </p>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
