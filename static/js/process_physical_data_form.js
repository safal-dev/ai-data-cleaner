// static/js/process_physical_data_form.js

document.addEventListener('DOMContentLoaded', () => {

    // --- File Upload Area Functionality for IMAGES ---
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const fileDisplayArea = document.getElementById('fileDisplayArea');

    if (dropArea && fileInput && fileDisplayArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
        });

        dropArea.addEventListener('drop', (e) => {
            const droppedFiles = e.dataTransfer.files;
            if (droppedFiles.length > 0) {
                fileInput.files = droppedFiles;
                displaySelectedImages(fileInput.files);
            }
        }, false);

        fileInput.addEventListener('change', () => {
            displaySelectedImages(fileInput.files);
        });

        function displaySelectedImages(files) {
            fileDisplayArea.innerHTML = '';
            if (files.length === 0) {
                fileDisplayArea.className = 'file-display-area';
                return;
            }

            fileDisplayArea.className = 'file-display-area image-preview-container';
            
            Array.from(files).forEach(file => {
                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function(event) {
                        const previewWrapper = document.createElement('div');
                        previewWrapper.classList.add('image-preview-wrapper');
                        previewWrapper.innerHTML = `
                            <img src="${event.target.result}" alt="${file.name}" class="image-preview">
                            <button type="button" class="remove-file-item" data-filename="${file.name}">×</button>
                        `;
                        fileDisplayArea.appendChild(previewWrapper);
                        
                        previewWrapper.querySelector('.remove-file-item').addEventListener('click', () => {
                            removeFileFromInput(file.name);
                        });
                    };
                    reader.readAsDataURL(file);
                }
            });
        }

        function removeFileFromInput(fileName) {
            const currentFiles = Array.from(fileInput.files);
            const newFiles = currentFiles.filter(file => file.name !== fileName);
            const dataTransfer = new DataTransfer();
            newFiles.forEach(file => dataTransfer.items.add(file));
            fileInput.files = dataTransfer.files;
            displaySelectedImages(fileInput.files);
        }
    }

    // --- Instruction Selection Modal Functionality ---
    const instructionModal = document.getElementById('instructionModal');
    const selectInstructionButton = document.getElementById('selectInstructionButton');
    const confirmButton = document.getElementById('confirmInstructionSelection');
    const selectedInstructionIdInput = document.getElementById('selectedInstructionId');
    const selectedInstructionNameSpan = document.getElementById('selectedInstructionName');

    if (instructionModal && selectInstructionButton && confirmButton && selectedInstructionIdInput && selectedInstructionNameSpan) {
        const closeButton = instructionModal.querySelector('.close-button');
        const cancelButton = instructionModal.querySelector('.modal-cancel-button');
        
        const openModal = () => instructionModal.style.display = 'flex';
        const closeModal = () => instructionModal.style.display = 'none';

        selectInstructionButton.addEventListener('click', openModal);
        closeButton.addEventListener('click', closeModal);
        cancelButton.addEventListener('click', closeModal);
        instructionModal.addEventListener('click', (event) => {
            if (event.target === instructionModal) closeModal();
        });

        confirmButton.addEventListener('click', () => {
            const selectedRadio = instructionModal.querySelector('input[name="instruction_radio"]:checked');
            if (selectedRadio) {
                selectedInstructionIdInput.value = selectedRadio.value;
                selectedInstructionNameSpan.textContent = selectedRadio.dataset.instructionName;
            }
            closeModal();
        });
    }


    const physicalForm = document.querySelector('.physical-data-form');
    const loadingOverlay = document.getElementById('loading-overlay');

    if (physicalForm && loadingOverlay) {
        physicalForm.addEventListener('submit', (event) => {
            // Optional: check if there's a file selected
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files.length) {
                alert("Please select an image to process.");
                event.preventDefault(); // Stop the form from submitting
                return;
            }
            loadingOverlay.classList.add('visible');
        });
    }

    
});




