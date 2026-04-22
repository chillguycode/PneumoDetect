// Get references to all the HTML elements we'll be interacting with
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const previewText = document.getElementById('previewText');
const predictButton = document.getElementById('predictButton');
const resultsDiv = document.getElementById('results');
const resultClass = document.getElementById('resultClass');
const resultConfidence = document.getElementById('resultConfidence');
const loader = document.getElementById('loader');

// Modal elements
const errorModal = document.getElementById('errorModal');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');
const modalCloseButton = document.getElementById('modalCloseButton');

//Heatmap elements
const pneumoCam = document.getElementById('pneumoCam');
const xrayOverlay = document.getElementById('xrayOverlay');
const heatmapOverlay = document.getElementById('heatmapOverlay');
const downloadBtn = document.getElementById('downloadBtn');
const mergeCanvas = document.getElementById('mergeCanvas');

// Define the URL of your FastAPI endpoint
//const API_ENDPOINT = 'https://pneumodetect-backend-720802368286.asia-south1.run.app/predict';

const API_ENDPOINT = window.location.hostname === 'localhost'
      ? '/predict'
      : 'https://pneumodetect-backend-720802368286.asia-south1.run.app/predict';

/**
 * Displays a custom message modal.
 * @param {string} title - The title of the message (e.g., "Error", "Guard Rejection").
 * @param {string} message - The main body of the message.
 */
function showMessageModal(title, message) {
    modalTitle.textContent = title;
    modalMessage.textContent = message;
    errorModal.classList.remove('hidden');
}

// Event listener to close the modal
modalCloseButton.addEventListener('click', () => {
    errorModal.classList.add('hidden');
});


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
	pneumoCam.classList.add('hidden');
	heatmapOverlay.classList.add('hidden');
	heatmapOverlay.src='';
	xrayOverlay.src='';

    }
});

// Add an event listener for when the predict button is clicked
predictButton.addEventListener('click', async () => {
    const file = imageUpload.files[0];
    if (!file) {
        showMessageModal("Upload Required", "Please select an image first!");
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

        // Wait for the JSON response from the server
        const data = await response.json();

        if (!response.ok) {
            // Handle specific HTTP status codes
            if (response.status === 406) {
                // Guard Model Rejection: 406 Not Acceptable
                showMessageModal("Not a Chest X-Ray", "This image doesn't appear to be a chest X-ray. Please upload a valid chest X-ray image.");
            } else if (response.status === 400) {
                // Bad Request (e.g., wrong file type)
                showMessageModal("Invalid Input (400)", data.detail);
            } else if (data.detail) {
                // General FastAPI error with a detail field
                showMessageModal(`Server Error (${response.status})`, data.detail);
            } else {
                // Generic network error
                throw new Error('Network response was not ok');
            }
            return; // Stop execution on error
        }

        // --- SUCCESS RESPONSE ---

        // Update the main prediction result elements
        resultClass.textContent = data.prediction;
        // Format confidence to a percentage
        const confidencePercentage = (data.confidence * 100).toFixed(2);
        resultConfidence.textContent = `${confidencePercentage}%`;
        
        // Show the results
        resultsDiv.classList.remove('hidden');
	
	// The CAM Heatmap
	xrayOverlay.src = `data:image/png;base64,${data.xray}`;
	heatmapOverlay.src = `data:image/png;base64,${data.heatmap}`;
	heatmapOverlay.classList.remove('hidden');
	pneumoCam.classList.remove('hidden');

    } catch (error) {
        console.error('Error:', error);
        showMessageModal("Connection Error", `An error occurred while making the prediction: ${error.message}`);
    } finally {
        // Hide the loader and re-enable the button
        loader.classList.add('hidden');
        predictButton.disabled = false;
    }
});


downloadBtn.addEventListener('click', () => {
	const ctx = mergeCanvas.getContext('2d');
	const img1 = new Image();
	const img2 = new Image();

	img1.onload = () => {
	  ctx.drawImage(img1, 0, 0, 224, 224);
	  img2.onload = () => {
	      ctx.globalAlpha = 0.5;
	      ctx.drawImage(img2, 0, 0, 224, 224);
	      ctx.globalAlpha = 1.0;

	      const link = document.createElement('a');
	      link.download = 'pneumocam.png';
	      link.href = mergeCanvas.toDataURL('image/png');
	      link.click();
	  };
	  img2.src = heatmapOverlay.src;
	};
	img1.src = xrayOverlay.src;
});
