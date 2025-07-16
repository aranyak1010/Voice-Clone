class CelebrityVoiceApp {
    constructor() {
        this.selectedVoice = null;
        this.selectedVoiceType = 'political';
        this.ethicalConsent = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.uploadedAudio = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadVoiceStyles();
        this.checkEthicalConsent();
    }

    setupEventListeners() {
        // Ethical consent
        document.getElementById('ethicalConsent').addEventListener('change', (e) => {
            this.ethicalConsent = e.target.checked;
            this.updateConvertButton();
        });

        // Voice type tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.selectVoiceType(e.target.dataset.type);
            });
        });

        // Audio upload
        const audioUploadArea = document.getElementById('audioUploadArea');
        const audioInput = document.getElementById('audioInput');
        
        audioUploadArea.addEventListener('click', () => audioInput.click());
        audioInput.addEventListener('change', this.handleAudioUpload.bind(this));

        // Recording
        document.getElementById('startRecording').addEventListener('click', this.startRecording.bind(this));
        document.getElementById('stopRecording').addEventListener('click', this.stopRecording.bind(this));

        // Voice conversion
        document.getElementById('convertVoice').addEventListener('click', this.convertVoice.bind(this));

        // Text input
        document.getElementById('textInput').addEventListener('input', this.updateConvertButton.bind(this));
    }

    checkEthicalConsent() {
        const consentCheckbox = document.getElementById('ethicalConsent');
        this.ethicalConsent = consentCheckbox.checked;
        this.updateConvertButton();
    }

    selectVoiceType(type) {
        this.selectedVoiceType = type;
        
        // Update tab appearance
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-type="${type}"]`).classList.add('active');

        // Load voice styles for selected type
        this.loadVoiceStyles();
    }

    async loadVoiceStyles() {
        try {
            let endpoint;
            if (this.selectedVoiceType === 'singing') {
                endpoint = '/voice_styles/singing';
            } else {
                endpoint = '/voice_styles/celebrity';
            }

            const response = await fetch(endpoint);
            const data = await response.json();

            if (response.ok) {
                this.displayVoiceStyles(data.styles);
            } else {
                this.showStatus(data.error || 'Error loading voice styles', 'error');
            }
        } catch (err) {
            this.showStatus('Error loading voice styles: ' + err.message, 'error');
        }
    }

    displayVoiceStyles(styles) {
        const grid = document.getElementById('celebrityGrid');
        grid.innerHTML = '';

        styles.forEach(style => {
            // Filter styles by current voice type
            if (this.selectedVoiceType === 'political' && style.type !== 'political') return;
            if (this.selectedVoiceType === 'entertainment' && style.type !== 'entertainment') return;
            if (this.selectedVoiceType === 'singing' && style.type !== 'singing') return;

            const card = document.createElement('div');
            card.className = 'celebrity-card';
            card.dataset.voiceId = style.id;
            
            let icon = '🎭';
            if (style.type === 'political') icon = '🏛️';
            if (style.type === 'entertainment') icon = '🎬';
            if (style.type === 'singing') icon = '🎵';

            card.innerHTML = `
                <div style="font-size: 2rem; margin-bottom: 10px;">${icon}</div>
                <h4>${style.name}</h4>
                <p style="font-size: 0.9em; color: #6c757d;">
                    ${style.type === 'singing' ? style.vocal_technique : 'Voice conversion'}
                </p>
                <div class="ethical-badge" style="font-size: 0.8em; color: #856404; margin-top: 8px;">
                    ⚠️ Educational use only
                </div>
            `;

            card.addEventListener('click', () => this.selectVoice(style.id, style.name));
            grid.appendChild(card);
        });

        if (styles.length === 0) {
            grid.innerHTML = '<p style="text-align: center; color: #6c757d;">No voice styles available for this category</p>';
        }
    }

    selectVoice(voiceId, voiceName) {
        this.selectedVoice = voiceId;

        // Update card appearance
        document.querySelectorAll('.celebrity-card').forEach(card => {
            card.classList.remove('selected');
        });
        document.querySelector(`[data-voice-id="${voiceId}"]`).classList.add('selected');

        this.showStatus(`Selected: ${voiceName}`, 'success');
        this.updateConvertButton();
    }

    async handleAudioUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.type.startsWith('audio/')) {
            this.showStatus('Please select an audio file', 'error');
            return;
        }

        this.uploadedAudio = file;
        this.showStatus(`Audio file uploaded: ${file.name}`, 'success');
        this.updateConvertButton();

        // Analyze if it's singing
        if (this.selectedVoiceType === 'singing') {
            await this.analyzeSinging(file);
        }
    }

    async analyzeSinging(audioFile) {
        try {
            const formData = new FormData();
            formData.append('audio', audioFile);

            const response = await fetch('/analyze_singing', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                const isSinging = data.is_singing;
                const recommended = data.recommended_singers;

                this.showStatus(
                    `Analysis: ${isSinging ? 'Singing detected' : 'Speech detected'}. 
                     Recommended style: ${recommended.join(', ')}`, 
                    'success'
                );
            }
        } catch (err) {
            console.warn('Singing analysis failed:', err);
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
            this.updateRecordingUI();

        } catch (err) {
            this.showStatus('Error starting recording: ' + err.message, 'error');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateRecordingUI();
        }
    }

    updateRecordingUI() {
        const startBtn = document.getElementById('startRecording');
        const stopBtn = document.getElementById('stopRecording');

        if (this.isRecording) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            startBtn.innerHTML = '🔴 Recording...';
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startBtn.innerHTML = '🎤 Record Audio';
        }
    }

    async processRecordedAudio() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], `recording_${Date.now()}.webm`, { type: 'audio/webm' });
        
        this.uploadedAudio = audioFile;
        this.showStatus('Voice recording completed', 'success');
        this.updateConvertButton();

        // Analyze if it's singing
        if (this.selectedVoiceType === 'singing') {
            await this.analyzeSinging(audioFile);
        }
    }

    async convertVoice() {
        if (!this.ethicalConsent) {
            this.showStatus('Please accept the ethical guidelines first', 'warning');
            return;
        }

        if (!this.selectedVoice) {
            this.showStatus('Please select a voice style first', 'warning');
            return;
        }

        const textInput = document.getElementById('textInput').value.trim();
        
        if (!this.uploadedAudio && !textInput) {
            this.showStatus('Please provide audio file or text input', 'warning');
            return;
        }

        const convertBtn = document.getElementById('convertVoice');
        const originalText = convertBtn.innerHTML;
        convertBtn.innerHTML = '<span class="loading"></span> Converting...';
        convertBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('target_voice', this.selectedVoice);
            formData.append('voice_type', this.selectedVoiceType === 'singing' ? 'singing' : 'celebrity');
            formData.append('ethical_consent', 'true');
            formData.append('use_case', 'educational_demonstration');

            if (this.uploadedAudio) {
                formData.append('audio', this.uploadedAudio);
            }

            if (textInput) {
                formData.append('text', textInput);
            }

            const response = await fetch('/convert_voice', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.displayConversionResult(data);
                this.showStatus('Voice conversion completed successfully!', 'success');
            } else {
                this.showStatus(data.error || 'Error during voice conversion', 'error');
            }

        } catch (err) {
            this.showStatus('Error during voice conversion: ' + err.message, 'error');
        } finally {
            convertBtn.innerHTML = originalText;
            convertBtn.disabled = false;
        }
    }

    displayConversionResult(data) {
        // Play converted audio
        const audioPlayer = document.getElementById('resultAudio');
        const audioBlob = this.base64ToBlob(data.audio_base64, 'audio/wav');
        const audioUrl = URL.createObjectURL(audioBlob);
        
        audioPlayer.src = audioUrl;
        audioPlayer.style.display = 'block';
        audioPlayer.play();

        // Show conversion details
        const conversionInfo = document.getElementById('conversionInfo');
        const conversionDetails = document.getElementById('conversionDetails');
        
        conversionDetails.innerHTML = `
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <p><strong>Conversion Type:</strong> ${data.conversion_type}</p>
                <p><strong>Target Voice:</strong> ${data.target_voice}</p>
                <p><strong>Voice Category:</strong> ${data.voice_type}</p>
                <p><strong>Sample Rate:</strong> ${data.sample_rate} Hz</p>
                <div style="margin-top: 10px; padding: 10px; background: #fff3cd; border-radius: 5px; font-size: 0.9em;">
                    ⚠️ ${data.ethical_notice}
                </div>
            </div>
        `;
        
        conversionInfo.style.display = 'block';
    }

    updateConvertButton() {
        const convertBtn = document.getElementById('convertVoice');
        const hasInput = this.uploadedAudio || document.getElementById('textInput').value.trim();
        
        convertBtn.disabled = !this.ethicalConsent || !this.selectedVoice || !hasInput;
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

// Initialize the celebrity voice app
document.addEventListener('DOMContentLoaded', () => {
    window.celebrityApp = new CelebrityVoiceApp();
});
