// Get references to all the HTML elements we'll be interacting with
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const previewText = document.getElementById('previewText');
const predictButton = document.getElementById('predictButton');
const resultsDiv = document.getElementById('results');
const resultClass = document.getElementById('resultClass');
const resultConfidence = document.getElementById('resultConfidence');
const loader = document.getElementById('loader');

// Define the URL of your FastAPI endpoint
const API_ENDPOINT = 'http://127.0.0.1:8000/predict';

// Add an event listener for when a file is selected
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        // Create a FileReader to read the image file
        const reader = new FileReader();
        
        reader.onload = (e) => {
            // When the file is loaded, set the src of the img tag
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            previewText.style.display = 'none';
            // Enable the predict button since we have an image
            predictButton.disabled = false;
        };
        
        // Read the file as a data URL (a base64-encoded string)
        reader.readAsDataURL(file);
        
        // Hide any previous results
        resultsDiv.classList.add('hidden');
    }
});

// Add an event listener for when the predict button is clicked
predictButton.addEventListener('click', async () => {
    const file = imageUpload.files[0];
    if (!file) {
        alert("Please select an image first!");
        return;
    }
    
    // Show the loader and hide results/button
    loader.classList.remove('hidden');
    resultsDiv.classList.add('hidden');
    predictButton.disabled = true;

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Use the fetch API to send the image to the backend
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        // Wait for the JSON response from the server
        const data = await response.json();

        // Update the result elements with the prediction data
        resultClass.textContent = data.class;
        // Format confidence to a percentage
        const confidencePercentage = (data.confidence * 100).toFixed(2);
        resultConfidence.textContent = `${confidencePercentage}%`;
        
        // Show the results
        resultsDiv.classList.remove('hidden');

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while making the prediction. Please check the console.');
    } finally {
        // Hide the loader and re-enable the button
        loader.classList.add('hidden');
        predictButton.disabled = false;
    }
});
