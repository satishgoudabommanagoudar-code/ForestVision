document.addEventListener("DOMContentLoaded", function () {

    const btn = document.getElementById("predictBtn");

    btn.addEventListener("click", async function () {

        const fileInput = document.getElementById("imageInput");
        const result = document.getElementById("result");

        if (fileInput.files.length === 0) {
            result.innerHTML = "Please select an image.";
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        result.innerHTML = "⏳ Predicting...";

        try {

            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

           alert("Prediction: " + data.prediction + "\nConfidence: " + data.confidence + "%");

alert("Prediction: " + data.prediction + "\nConfidence: " + data.confidence + "%");

result.textContent =
    "Prediction: " + data.prediction +
    " | Confidence: " + data.confidence + "%";

result.style.display = "block";
result.style.background = "yellow";
result.style.color = "black";
result.style.padding = "20px";
result.style.fontSize = "24px";
result.style.border = "2px solid black";
        } catch (error) {

            console.error(error);

        result.innerHTML = `
<h2>📋 Classification Result</h2>

<p><strong>Class:</strong> ${data.prediction}</p>

<p><strong>Confidence:</strong> ${data.confidence}%</p>
`;
        }

    });

});