// static/js/process_data_form.js

document.addEventListener('DOMContentLoaded', () => {

    // --- File Upload Area Functionality ---
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const fileDisplayArea = document.getElementById('fileDisplayArea');

    // Make sure all elements exist before adding listeners
    if (dropArea && fileInput && fileDisplayArea) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // Highlight drop area on drag enter/over
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
        });

        // Remove highlight on drag leave/drop
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
        });

        // Handle files dropped into the drop area
        dropArea.addEventListener('drop', (e) => {
            const droppedFiles = e.dataTransfer.files;
            if (droppedFiles.length > 0) {
                fileInput.files = droppedFiles;
                displaySelectedFiles(fileInput.files);
            }
        }, false);

        // Handle file selection from the browse button
        fileInput.addEventListener('change', () => {
            displaySelectedFiles(fileInput.files);
        });

        // Unified file display function
        function displaySelectedFiles(files) {
            fileDisplayArea.innerHTML = '';
            if (files.length === 0) {
                return;
            }

            const ul = document.createElement('ul');
            ul.classList.add('file-list');
            Array.from(files).forEach(file => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span class="file-name">${file.name}</span>
                    <button type="button" class="remove-file-item" data-filename="${file.name}">×</button>
                `;
                ul.appendChild(li);
                // Attach event listener for the new button
                li.querySelector('.remove-file-item').addEventListener('click', (event) => {
                    const fileNameToRemove = event.target.dataset.filename;
                    removeFileFromInput(fileNameToRemove);
                });
            });
            fileDisplayArea.appendChild(ul);
        }

        // Function to remove a file from the actual fileInput
        function removeFileFromInput(fileName) {
            const currentFiles = Array.from(fileInput.files);
            const newFiles = currentFiles.filter(file => file.name !== fileName);

            // The DataTransfer object is the standard way to programmatically update a file input's list
            const dataTransfer = new DataTransfer();
            newFiles.forEach(file => dataTransfer.items.add(file));
            fileInput.files = dataTransfer.files;

            // Update the UI to reflect the change
            displaySelectedFiles(fileInput.files);
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
        const instructionRadios = instructionModal.querySelectorAll('input[name="instruction_radio"]');
        
        const openModal = () => {
            instructionModal.style.display = 'flex';
            // Pre-select the currently active instruction in the modal
            const currentSelectedId = selectedInstructionIdInput.value;
            if (currentSelectedId) {
                const radioToSelect = instructionModal.querySelector(`input[value="${currentSelectedId}"]`);
                if (radioToSelect) radioToSelect.checked = true;
            }
        };

        const closeModal = () => {
            instructionModal.style.display = 'none';
        };

        selectInstructionButton.addEventListener('click', openModal);
        closeButton.addEventListener('click', closeModal);
        cancelButton.addEventListener('click', closeModal);

        // Close modal if user clicks outside the content area
        instructionModal.addEventListener('click', (event) => {
            if (event.target === instructionModal) {
                closeModal();
            }
        });

        // Handle the confirmation
        confirmButton.addEventListener('click', () => {
            const selectedRadio = instructionModal.querySelector('input[name="instruction_radio"]:checked');
            if (selectedRadio) {
                selectedInstructionIdInput.value = selectedRadio.value;
                selectedInstructionNameSpan.textContent = selectedRadio.dataset.instructionName;
            }
            closeModal();
        });
    }
});