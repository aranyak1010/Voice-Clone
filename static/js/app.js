class VoiceCloneApp {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.recordingTimer = null;
        this.recordingStartTime = 0;
        this.maxRecordingTime = 30000; // 30 seconds
        this.uploadedFiles = [];
        this.selectedSpeaker = null;
        this.speakers = [];
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadSpeakers();
        this.checkMicrophonePermission();
    }

    setupEventListeners() {
        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Audio recording
        document.getElementById('startRecording').addEventListener('click', this.startRecording.bind(this));
        document.getElementById('stopRecording').addEventListener('click', this.stopRecording.bind(this));
        
        // Voice synthesis
        document.getElementById('synthesizeBtn').addEventListener('click', this.synthesizeText.bind(this));
        
        // Upload and train
        document.getElementById('uploadTrainBtn').addEventListener('click', this.uploadAndTrain.bind(this));
    }

    // File Upload Handlers
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        this.processFiles(files);
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.processFiles(files);
    }

    processFiles(files) {
        const audioFiles = files.filter(file => 
            file.type.startsWith('audio/') || file.name.endsWith('.wav') || file.name.endsWith('.mp3')
        );
        
        if (audioFiles.length === 0) {
            this.showStatus('Please select audio files (WAV, MP3)', 'error');
            return;
        }

        this.uploadedFiles = [...this.uploadedFiles, ...audioFiles];
        this.displayUploadedFiles();
        this.showStatus(`${audioFiles.length} file(s) added successfully`, 'success');
    }

    displayUploadedFiles() {
        const fileList = document.getElementById('fileList');
        fileList.innerHTML = '';

        this.uploadedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span class="file-name">${file.name}</span>
                <span class="file-size">${this.formatFileSize(file.size)}</span>
                <button class="btn btn-danger" onclick="app.removeFile(${index})">Remove</button>
            `;
            fileList.appendChild(fileItem);
        });

        document.getElementById('uploadTrainBtn').disabled = this.uploadedFiles.length === 0;
    }

    removeFile(index) {
        this.uploadedFiles.splice(index, 1);
        this.displayUploadedFiles();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Audio Recording
    async checkMicrophonePermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            document.getElementById('microphoneStatus').textContent = 'Microphone ready';
            document.getElementById('startRecording').disabled = false;
        } catch (err) {
            document.getElementById('microphoneStatus').textContent = 'Microphone access denied';
            this.showStatus('Microphone access is required for voice recording', 'error');
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    sampleRate: 22050,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });

            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            this.audioChunks = [];
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                this.processRecordedAudio();
                stream.getTracks().forEach(track => track.stop());
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            this.recordingStartTime = Date.now();

            this.updateRecordingUI();
            this.startRecordingTimer();

            // Auto-stop after 30 seconds
            setTimeout(() => {
                if (this.isRecording) {
                    this.stopRecording();
                }
            }, this.maxRecordingTime);

        } catch (err) {
            this.showStatus('Error starting recording: ' + err.message, 'error');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateRecordingUI();
            this.stopRecordingTimer();
        }
    }

    updateRecordingUI() {
        const startBtn = document.getElementById('startRecording');
        const stopBtn = document.getElementById('stopRecording');
        const indicator = document.getElementById('recordingIndicator');

        if (this.isRecording) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            indicator.style.display = 'flex';
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            indicator.style.display = 'none';
        }
    }

    startRecordingTimer() {
        this.recordingTimer = setInterval(() => {
            const elapsed = Date.now() - this.recordingStartTime;
            const remaining = Math.max(0, this.maxRecordingTime - elapsed);
            
            document.getElementById('timer').textContent = 
                this.formatTime(remaining);

            const progress = (elapsed / this.maxRecordingTime) * 100;
            document.getElementById('recordingProgress').style.width = `${Math.min(progress, 100)}%`;

        }, 100);
    }

    stopRecordingTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
        document.getElementById('recordingProgress').style.width = '0%';
        document.getElementById('timer').textContent = '00:30';
    }

    formatTime(ms) {
        const seconds = Math.ceil(ms / 1000);
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    async processRecordedAudio() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        
        // Convert to WAV format
        const audioBuffer = await this.convertToWav(audioBlob);
        const wavBlob = new Blob([audioBuffer], { type: 'audio/wav' });
        
        // Add to uploaded files
        const file = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        this.uploadedFiles.push(file);
        this.displayUploadedFiles();
        
        this.showStatus('Voice recording added successfully', 'success');
    }

    async convertToWav(audioBlob) {
        // Simple WAV conversion (you might want to use a library like lamejs for better conversion)
        const arrayBuffer = await audioBlob.arrayBuffer();
        // This is a simplified conversion - in production, use proper audio conversion
        return arrayBuffer;
    }

    // Speaker Management
    async loadSpeakers() {
        try {
            const response = await fetch('/speakers');
            const data = await response.json();
            this.speakers = data.speakers || [];
            this.displaySpeakers();
        } catch (err) {
            this.showStatus('Error loading speakers: ' + err.message, 'error');
        }
    }

    displaySpeakers() {
        const speakerGrid = document.getElementById('speakerGrid');
        speakerGrid.innerHTML = '';

        this.speakers.forEach(speaker => {
            const speakerCard = document.createElement('div');
            speakerCard.className = 'speaker-card';
            speakerCard.innerHTML = `
                <h4>${speaker}</h4>
                <p>Click to select</p>
            `;
            speakerCard.addEventListener('click', () => this.selectSpeaker(speaker));
            speakerGrid.appendChild(speakerCard);
        });
    }

    async selectSpeaker(speakerId) {
        try {
            const response = await fetch(`/speaker/${speakerId}`, { method: 'POST' });
            const data = await response.json();
            
            if (response.ok) {
                this.selectedSpeaker = speakerId;
                this.updateSpeakerSelection();
                this.showStatus(`Speaker ${speakerId} selected`, 'success');
            } else {
                this.showStatus(data.error || 'Error selecting speaker', 'error');
            }
        } catch (err) {
            this.showStatus('Error selecting speaker: ' + err.message, 'error');
        }
    }

    updateSpeakerSelection() {
        document.querySelectorAll('.speaker-card').forEach(card => {
            card.classList.remove('active');
        });
        
        if (this.selectedSpeaker) {
            const selectedCard = Array.from(document.querySelectorAll('.speaker-card'))
                .find(card => card.querySelector('h4').textContent === this.selectedSpeaker);
            if (selectedCard) {
                selectedCard.classList.add('active');
            }
        }
    }

    // Voice Synthesis
    async synthesizeText() {
        const textInput = document.getElementById('textInput');
        const text = textInput.value.trim();
        
        if (!text) {
            this.showStatus('Please enter text to synthesize', 'warning');
            return;
        }
        
        if (!this.selectedSpeaker) {
            this.showStatus('Please select a speaker first', 'warning');
            return;
        }

        const synthesizeBtn = document.getElementById('synthesizeBtn');
        const originalText = synthesizeBtn.innerHTML;
        synthesizeBtn.innerHTML = '<span class="loading"></span> Synthesizing...';
        synthesizeBtn.disabled = true;

        try {
            const response = await fetch('/synthesize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, format: 'base64' })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.playGeneratedAudio(data.audio_base64);
                this.showStatus('Speech synthesized successfully', 'success');
            } else {
                this.showStatus(data.error || 'Error synthesizing speech', 'error');
            }
        } catch (err) {
            this.showStatus('Error synthesizing speech: ' + err.message, 'error');
        } finally {
            synthesizeBtn.innerHTML = originalText;
            synthesizeBtn.disabled = false;
        }
    }

    playGeneratedAudio(audioBase64) {
        const audioPlayer = document.getElementById('audioPlayer');
        const audioBlob = this.base64ToBlob(audioBase64, 'audio/wav');
        const audioUrl = URL.createObjectURL(audioBlob);
        
        audioPlayer.src = audioUrl;
        audioPlayer.style.display = 'block';
        audioPlayer.play();
    }

    base64ToBlob(base64, mimeType) {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }

    // Upload and Training
    async uploadAndTrain() {
        if (this.uploadedFiles.length === 0) {
            this.showStatus('Please add audio files first', 'warning');
            return;
        }

        const speakerName = document.getElementById('speakerName').value.trim();
        if (!speakerName) {
            this.showStatus('Please enter a speaker name', 'warning');
            return;
        }

        const uploadBtn = document.getElementById('uploadTrainBtn');
        const originalText = uploadBtn.innerHTML;
        uploadBtn.innerHTML = '<span class="loading"></span> Training...';
        uploadBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('speaker_id', speakerName);
            
            this.uploadedFiles.forEach((file, index) => {
                formData.append('audio_files', file);
            });

            const response = await fetch('/upload_and_train', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                this.showStatus('Voice model trained successfully!', 'success');
                this.uploadedFiles = [];
                this.displayUploadedFiles();
                this.loadSpeakers(); // Refresh speaker list
                document.getElementById('speakerName').value = '';
            } else {
                this.showStatus(data.error || 'Error training voice model', 'error');
            }
        } catch (err) {
            this.showStatus('Error uploading and training: ' + err.message, 'error');
        } finally {
            uploadBtn.innerHTML = originalText;
            uploadBtn.disabled = false;
        }
    }

    showStatus(message, type) {
        const statusDiv = document.getElementById('statusMessage');
        statusDiv.className = `status-message status-${type}`;
        statusDiv.textContent = message;
        statusDiv.classList.remove('hidden');
        
        setTimeout(() => {
            statusDiv.classList.add('hidden');
        }, 5000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new VoiceCloneApp();
});
