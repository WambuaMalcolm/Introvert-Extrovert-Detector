document
  .getElementById("predictForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = {};

    formData.forEach((value, key) => {
      // Convert only numeric fields to numbers
      if (
        [
          "Time_spent_Alone",
          "Social_event_attendance",
          "Going_outside",
          "Friends_circle_size",
          "Post_frequency",
        ].includes(key)
      ) {
        data[key] = Number(value);
      } else {
        data[key] = value; // Keep strings for categorical features
      }
    });

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      document.getElementById("result").innerHTML = `
        <div class="result-box">
          <strong>Prediction:</strong> ${result.prediction}<br>
          <strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%
        </div>
      `;
    } catch (error) {
      document.getElementById("result").textContent =
        "⚠️ Error: Unable to fetch prediction.";
    }
  });
