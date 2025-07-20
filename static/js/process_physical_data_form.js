 document.addEventListener('DOMContentLoaded', () => {
            // Hamburger menu functionality (same as before)
            const hamburgerMenu = document.querySelector('.hamburger-menu');
            const navLinks = document.querySelector('.nav-links');

            if (hamburgerMenu && navLinks) {
                hamburgerMenu.addEventListener('click', () => {
                    navLinks.classList.toggle('active');
                    hamburgerMenu.classList.toggle('active');
                });
            }

            // --- File Upload Area Functionality ---
            const dropArea = document.getElementById('dropArea');
            const fileInput = document.getElementById('fileInput');
            const fileDisplayArea = document.getElementById('fileDisplayArea');

            // Hardcode dataType for physical
            const dataType = 'physical'; 

            // Prevent default drag behaviors for the whole document and dropArea
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            // Highlight drop area on drag enter/over
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false);
            });

            // Remove highlight on drag leave/drop
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false);
            });

            function highlight() { dropArea.classList.add('highlight'); }
            function unhighlight() { dropArea.classList.remove('highlight'); }

            // Handle files dropped into the drop area (JavaScript way)
            dropArea.addEventListener('drop', (e) => {
                preventDefaults(e); 
                unhighlight();      

                const droppedFiles = e.dataTransfer.files;

                if (droppedFiles && droppedFiles.length > 0) {
                    console.log('DEBUG (Physical): Drop event detected. Dropped files count:', droppedFiles.length, 'Names:', Array.from(droppedFiles).map(f => f.name));
                    // Directly assign the dropped files to the file input
                    fileInput.files = droppedFiles; 
                    displaySelectedFiles(fileInput.files); // Call with current files from input
                } else {
                    console.log('DEBUG (Physical): No files dropped or invalid drop.');
                    fileInput.files = new DataTransfer().files; // Clear the input if empty drop
                    displaySelectedFiles(fileInput.files); // Clear display
                }
            }, false);

            // Handle file selection from browse button (native HTML input 'change' event)
            fileInput.addEventListener('change', (event) => {
                event.stopPropagation();

                const selectedFiles = event.target.files;
                console.log('DEBUG (Physical): fileInput change event triggered. Selected files count:', selectedFiles.length, 'Names:', Array.from(selectedFiles).map(f => f.name));

                if (!selectedFiles || selectedFiles.length === 0) {
                    console.log('DEBUG (Physical): Change event fired with no files or cleared selection. Clearing display.');
                    displaySelectedFiles(new DataTransfer().files); // Clear display by passing empty FileList
                } else {
                    displaySelectedFiles(selectedFiles); // Direct call
                }
            });


            // --- MODIFIED: UNIFIED FILE DISPLAY FUNCTION ---
            // Removed 'type' parameter as it's now specific to physical_data
            function displaySelectedFiles(files) {
                console.log("DEBUG (Physical): displaySelectedFiles called. Files count:", files.length, "Files for display:", Array.from(files).map(f => f.name));
                
                fileDisplayArea.innerHTML = ''; // Crucial: Clear existing content completely
                console.log("DEBUG (Physical): fileDisplayArea cleared. Current HTML length after clear:", fileDisplayArea.innerHTML.length);

                if (files.length === 0) {
                    fileDisplayArea.classList.remove('image-preview-container'); // Ensure correct class is removed
                    console.log("DEBUG (Physical): No files, display area classes removed. Exiting displaySelectedFiles.");
                    return;
                }

                fileDisplayArea.classList.add('image-preview-container'); // Ensure this class is always present for physical previews

                Array.from(files).forEach(file => {
                    console.log("DEBUG (Physical): Processing file for display:", file.name, "File type:", file.type);
                    
                    const previewItem = document.createElement('div');
                    previewItem.classList.add('image-preview-item');

                    // Check if it's an image for actual preview, otherwise show generic icon
                    if (file.type.startsWith('image/')) {
                        const reader = new FileReader();
                        reader.onload = (e) => {
                            previewItem.innerHTML = `
                                <img src="${e.target.result}" alt="${file.name}">
                                <span>${file.name}</span>
                                <button type="button" class="remove-file-item" data-filename="${file.name}">&times;</button>
                            `;
                            // Attach event listener immediately after innerHTML is set
                            attachRemoveButtonListener(previewItem.querySelector('.remove-file-item'));
                        };
                        reader.readAsDataURL(file);
                    } else {
                        // For non-image files (shouldn't happen often with accept="image/*" but good for robustness)
                        previewItem.classList.add('non-image');
                        previewItem.innerHTML = `
                            <i class="fas fa-file-alt"></i> {# Generic file icon for non-images #}
                            <span>${file.name}</span>
                            <p class="error-text">Not an image</p>
                            <button type="button" class="remove-file-item" data-filename="${file.name}">&times;</button>
                        `;
                        // Attach event listener immediately
                        attachRemoveButtonListener(previewItem.querySelector('.remove-file-item'));
                    }
                    fileDisplayArea.appendChild(previewItem);
                });

                console.log("DEBUG (Physical): fileDisplayArea content after update. Final HTML length:", fileDisplayArea.innerHTML.length, "Content exists:", fileDisplayArea.innerHTML.length > 0);
            }

            // Function to attach listener to a single remove button
            function attachRemoveButtonListener(button) {
                if (button) {
                    button.addEventListener('click', (event) => {
                        console.log('DEBUG (Physical): Remove button clicked.');
                        const fileNameToRemove = event.target.dataset.filename;
                        console.log('DEBUG (Physical): Attempting to remove file:', fileNameToRemove);
                        removeFileFromInput(fileNameToRemove);
                    });
                } else {
                    console.warn('WARNING (Physical): Tried to attach remove listener but button element not found.');
                }
            }

            // Function to remove a file from the actual fileInput.files
            function removeFileFromInput(fileName) {
                const currentFiles = Array.from(fileInput.files);
                console.log('DEBUG (Physical): removeFileFromInput: Current files count:', currentFiles.length, 'File to remove:', fileName);
                
                // Filter out the file to be removed
                const newFiles = currentFiles.filter(file => file.name !== fileName);
                console.log('DEBUG (Physical): removeFileFromInput: Files after filter (newFiles count):', newFiles.length);

                // Create a new DataTransfer object to update fileInput.files
                const dataTransfer = new DataTransfer();
                newFiles.forEach(file => {
                    dataTransfer.items.add(file);
                });
                fileInput.files = dataTransfer.files;
                console.log('DEBUG (Physical): removeFileFromInput: fileInput.files updated. New count:', fileInput.files.length);

                // Re-render the display area with the updated file list
                displaySelectedFiles(fileInput.files);
                console.log('DEBUG (Physical): removeFileFromInput: displaySelectedFiles called to refresh UI.');
            }


            // --- Instruction Selection Modal Functionality (Same as before) ---
            const instructionModal = document.getElementById('instructionModal');
            const selectInstructionButton = document.getElementById('selectInstructionButton');
            const closeButton = instructionModal.querySelector('.close-button');
            const cancelButton = instructionModal.querySelector('.modal-cancel-button');
            const confirmButton = document.getElementById('confirmInstructionSelection');
            const selectedInstructionIdInput = document.getElementById('selectedInstructionId');
            const selectedInstructionNameSpan = document.getElementById('selectedInstructionName');
            const instructionRadios = instructionModal.querySelectorAll('input[name="instruction_radio"]');

            function openModal() {
                instructionModal.style.display = 'flex';
                const currentSelectedId = selectedInstructionIdInput.value;
                if (currentSelectedId) {
                    const radioToSelect = instructionModal.querySelector(`input[value="${currentSelectedId}"]`);
                    if (radioToSelect) {
                        radioToSelect.checked = true;
                    }
                } else {
                    const defaultRadio = instructionModal.querySelector('input[name="instruction_radio"][checked]');
                    if (defaultRadio) {
                        defaultRadio.checked = true;
                    } else {
                        if (instructionRadios.length > 0) {
                            instructionRadios[0].checked = true;
                        }
                    }
                }
            }

            function closeModal() {
                instructionModal.style.display = 'none';
            }

            selectInstructionButton.addEventListener('click', openModal);
            closeButton.addEventListener('click', closeModal);
            cancelButton.addEventListener('click', closeModal);

            instructionModal.addEventListener('click', (event) => {
                if (event.target === instructionModal) {
                    closeModal();
                }
            });

            confirmButton.addEventListener('click', () => {
                let selectedRadio = null;
                instructionRadios.forEach(radio => {
                    if (radio.checked) {
                        selectedRadio = radio;
                    }
                });

                if (selectedRadio) {
                    selectedInstructionIdInput.value = selectedRadio.value;
                    selectedInstructionNameSpan.textContent = selectedRadio.dataset.instructionName;
                } else {
                    const initialDefaultName = "{{ default_instruction.name|default:'(None Selected - using default)' }}";
                    const initialDefaultPk = "{{ default_instruction.pk|default:'' }}";

                    selectedInstructionIdInput.value = initialDefaultPk;
                    selectedInstructionNameSpan.textContent = initialDefaultName;
                }
                closeModal();
                console.log("DEBUG (Physical): Instruction confirmed. ID:", selectedInstructionIdInput.value, "Name:", selectedInstructionNameSpan.textContent);
            });

            // Initial display of the default instruction name on page load
            const initialDefaultName = "{{ default_instruction.name|default:'(None Selected - using default)' }}";
            const currentSelectedIdValue = selectedInstructionIdInput.value;
            if (currentSelectedIdValue) {
                const defaultRadio = document.querySelector(`input[name="instruction_radio"][value="${currentSelectedIdValue}"]`);
                if (defaultRadio) {
                    selectedInstructionNameSpan.textContent = defaultRadio.dataset.instructionName;
                } else {
                    selectedInstructionNameSpan.textContent = initialDefaultName;
                }
            } else {
                 selectedInstructionNameSpan.textContent = initialDefaultName;
            }
            console.log("DEBUG (Physical): Initial instruction display set to:", selectedInstructionNameSpan.textContent);
        });