const steps = Array.from(document.querySelectorAll(".step-card"));
const form = document.getElementById("riskForm");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const stepLabel = document.getElementById("stepLabel");
const stepTitle = document.getElementById("stepTitle");
const progressFill = document.getElementById("progressFill");
const submitStatus = document.getElementById("submitStatus");
const resultsNote = document.getElementById("resultsNote");
const resultCards = document.getElementById("resultCards");
const presetSelects = Array.from(document.querySelectorAll(".preset-select"));

let currentStep = 0;

function updateStepState() {
    steps.forEach((step, index) => {
        step.classList.toggle("active", index === currentStep);
    });

    const activeStep = steps[currentStep];
    stepLabel.textContent = `Step ${currentStep + 1} of ${steps.length}`;
    stepTitle.textContent = activeStep.dataset.title;
    progressFill.style.width = `${((currentStep + 1) / steps.length) * 100}%`;
    prevBtn.disabled = currentStep === 0;
    nextBtn.style.display = currentStep === steps.length - 1 ? "none" : "inline-flex";
}

function validateCurrentStep() {
    const currentInputs = steps[currentStep].querySelectorAll("input, select");
    for (const input of currentInputs) {
        if (!input.reportValidity()) {
            input.focus();
            return false;
        }
    }
    return true;
}

function normalizePayload(formData) {
    const payload = {};
    const sharedBloodPressure = formData.get("shared_bp");
    const ckdBloodPressure = formData.get("ckd_bp");
    const sharedCholesterol = formData.get("shared_cholesterol");

    for (const [key, rawValue] of formData.entries()) {
        if (
            key === "patient_name" ||
            key === "shared_bp" ||
            key === "ckd_bp" ||
            key === "shared_cholesterol"
        ) {
            continue;
        }

        if (rawValue === "") {
            continue;
        }

        const numericKeys = new Set([
            "age",
            "sex",
            "bmi",
            "cigsperday",
            "bpmeds",
            "heartrate",
            "cp",
            "thalach",
            "exang",
            "thal",
            "glucose",
            "htn",
            "dm",
            "appet",
        ]);

        payload[key] = numericKeys.has(key) ? Number(rawValue) : rawValue;
    }

    if (typeof payload.sex === "number") {
        payload.male = payload.sex;
    }

    if (sharedBloodPressure !== null && sharedBloodPressure !== "") {
        const normalizedBloodPressure = Number(sharedBloodPressure);
        payload.trestbps = normalizedBloodPressure;
        payload.bloodpressure = normalizedBloodPressure;
    }

    if (ckdBloodPressure !== null && ckdBloodPressure !== "") {
        payload.bp = Number(ckdBloodPressure);
    }

    if (sharedCholesterol !== null && sharedCholesterol !== "") {
        const normalizedCholesterol = Number(sharedCholesterol);
        payload.chol = normalizedCholesterol;
        payload.totchol = normalizedCholesterol;
    }

    return payload;
}

function prettifyDiseaseName(name) {
    if (name === "ckd") {
        return "Chronic Kidney Disease";
    }
    return name.charAt(0).toUpperCase() + name.slice(1);
}

function buildResultCard(disease, result) {
    const card = document.createElement("article");
    const isPositive = Boolean(result.risk_detected);

    card.className = `result-card ${isPositive ? "positive" : "negative"}`;

    const probabilityItems = result.class_probabilities
        ? Object.entries(result.class_probabilities)
            .map(
                ([classLabel, probability]) =>
                    `<li><strong>${classLabel}</strong>: ${(probability * 100).toFixed(1)}%</li>`
            )
            .join("")
        : "<li>Probability output not available for this model.</li>";

    card.innerHTML = `
        <span class="result-chip">${isPositive ? "Higher risk pattern" : "Lower risk pattern"}</span>
        <h4>${prettifyDiseaseName(disease)}</h4>
        <p class="results-note">Model reading: <strong>${result.risk_label.replaceAll("_", " ")}</strong></p>
        <ul class="probability-list">${probabilityItems}</ul>
    `;

    return card;
}

function buildSkippedCard(disease, reason) {
    const card = document.createElement("article");
    card.className = "result-card skipped";
    card.innerHTML = `
        <span class="result-chip">Skipped</span>
        <h4>${prettifyDiseaseName(disease)}</h4>
        <p class="results-note">${reason}</p>
    `;
    return card;
}

async function submitForm(event) {
    event.preventDefault();

    if (!validateCurrentStep()) {
        return;
    }

    const formData = new FormData(form);
    const payload = normalizePayload(formData);
    const patientName = formData.get("patient_name") || "Patient";

    submitStatus.textContent = "Running all four models...";
    resultCards.innerHTML = "";

    try {
        const response = await fetch("/predict/all", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction request failed.");
        }

        resultsNote.textContent = `${patientName}'s screening results are ready.`;

        Object.entries(data.predictions || {}).forEach(([disease, result]) => {
            resultCards.appendChild(buildResultCard(disease, result));
        });

        Object.entries(data.skipped || {}).forEach(([disease, reason]) => {
            resultCards.appendChild(buildSkippedCard(disease, reason));
        });

        submitStatus.textContent = "Predictions generated successfully.";
    } catch (error) {
        submitStatus.textContent = error.message;
        resultsNote.textContent = "The request could not be completed.";
    }
}

prevBtn.addEventListener("click", () => {
    if (currentStep > 0) {
        currentStep -= 1;
        updateStepState();
        window.scrollTo({ top: 0, behavior: "smooth" });
    }
});

nextBtn.addEventListener("click", () => {
    if (!validateCurrentStep()) {
        return;
    }

    if (currentStep < steps.length - 1) {
        currentStep += 1;
        updateStepState();
        window.scrollTo({ top: 0, behavior: "smooth" });
    }
});

form.addEventListener("submit", submitForm);

presetSelects.forEach((selectElement) => {
    selectElement.addEventListener("change", () => {
        const targetField = form.querySelector(
            `[name="${selectElement.dataset.target}"]`
        );

        if (!targetField || !selectElement.value) {
            return;
        }

        targetField.value = selectElement.value;
    });
});

updateStepState();
