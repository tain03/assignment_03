document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    const predictionDiv = document.querySelector(".prediction");

    form.addEventListener("submit", function(event) {
        event.preventDefault();

        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });

        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            predictionDiv.innerHTML = `<h2>${result.prediction_text}</h2>`;
        })
        .catch(error => {
            console.error("Error:", error);
        });
    });
});