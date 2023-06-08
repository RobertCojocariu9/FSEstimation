const rgbInput = document.getElementById('rgb-image');
const depthInput = document.getElementById('depth-image');
const rgbPreview = document.getElementById('rgb-preview');
const depthPreview = document.getElementById('depth-preview');

// Event listeners to show image previews
rgbInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = (event) => {
        rgbPreview.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

depthInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = (event) => {
        depthPreview.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

const submitBtn = document.getElementById('submit-btn');
const resultDiv = document.getElementById('result-div');

// Form submission handle
const handleFormSubmit = (event) => {
    event.preventDefault();
    resultDiv.innerHTML = "";

    const rgbImage = document.getElementById('rgb-image').files[0];
    const depthImage = document.getElementById('depth-image').files[0];
    const fx = document.getElementById('fx-input').value;
    const cx = document.getElementById('cx-input').value;
    const fy = document.getElementById('fy-input').value;
    const cy = document.getElementById('cy-input').value;

    // Form validations
    if (!rgbImage || !depthImage || fx === '' || cx === '' || fy === '' || cy === '') {
        alert('Please upload both images and fill in the intrinsic parameters.');
        return;
    }

    if (!Number.isInteger(Number(fx)) && !Number.parseFloat(fx)) {
        alert('The value of fx must be an integer or a float.');
        return;
    }
    if (!Number.isInteger(Number(cx)) && !Number.parseFloat(cx)) {
        alert('The value of cx must be an integer or a float.');
        return;
    }
    if (!Number.isInteger(Number(fy)) && !Number.parseFloat(fy)) {
        alert('The value of fy must be an integer or a float.');
        return;
    }
    if (!Number.isInteger(Number(cy)) && !Number.parseFloat(cy)) {
        alert('The value of cy must be an integer or a float.');
        return;
    }

    const formData = new FormData();
    formData.append('file1', rgbImage);
    formData.append('file2', depthImage);
    formData.append('fx', fx);
    formData.append('cx', cx);
    formData.append('fy', fy);
    formData.append('cy', cy);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/predict', true);
    xhr.setRequestHeader('Access-Control-Allow-Origin', '*');
    xhr.responseType = 'json';
    xhr.onload = () => {
        if (xhr.status === 200) {
            // Decode base64 string
            const img1Data = xhr.response.img1;
            const img1Src = 'data:image/jpeg;base64,' + img1Data;
            const img1 = document.createElement('img');
            const img1Label = document.createElement('h2');
            img1Label.textContent = 'Freespace prediction';
            resultDiv.appendChild(img1Label);
            resultDiv.appendChild(img1);
            img1.src = img1Src;
        } else {
            const err = document.createElement('h2');
            err.textContent = 'An error occcured. Try again later.';
        }
    };
    xhr.send(formData);
};
submitBtn.addEventListener('click', handleFormSubmit);