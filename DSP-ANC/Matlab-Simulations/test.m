%% Real-Time Stereo ANC with Mono Mic + GUI Buttons
clc; clear; close all;

%% --- Load WAV file ---
[signal, fs] = audioread('BarreleyeFish.wav');  % Stereo WAV
signal = signal / max(abs(signal(:)));          % Normalize both channels
numChannels = size(signal,2);

%% --- Audio device setup ---
frameSize = 512;
micReader = audioDeviceReader('SampleRate',fs,'SamplesPerFrame',frameSize);
speakerWriter = audioDeviceWriter('SampleRate',fs);

%% --- ANC Parameters ---
filterOrder = 64;
mu = 0.001;
w = zeros(filterOrder, numChannels);  % Separate LMS weights for each stereo channel

%% --- Global control variables ---
global anc_enabled running
anc_enabled = false;
running = true;

%% --- GUI Setup ---
hFig = figure('Name','Stereo ANC (Mono Mic)','NumberTitle','off');
set(hFig,'CloseRequestFcn',@quitFcn);

% Toggle ANC button
uicontrol('Style','pushbutton','String','Toggle ANC',...
    'Position',[20 20 100 40], 'Callback',@toggleANC);

% Quit button
uicontrol('Style','pushbutton','String','Quit',...
    'Position',[140 20 100 40], 'Callback',@quitANC);

% Status text
statusText = uicontrol('Style','text','String','ANC: OFF',...
    'Position',[260 20 120 40]);

%% --- Main real-time loop ---
audioPointer = 1;
numSamples = size(signal,1);

disp('Click buttons to control ANC.');

while running && ishandle(hFig)
    % --- 1. Get next audio chunk ---
    if audioPointer + frameSize - 1 > numSamples
        chunk = signal(audioPointer:end, :);
        audioPointer = 1;
        chunk(end+1:frameSize, :) = 0;
    else
        chunk = signal(audioPointer:audioPointer+frameSize-1, :);
        audioPointer = audioPointer + frameSize;
    end
    
    % --- 2. Read microphone (mono) ---
    micData = micReader();
    micData = micData / (max(abs(micData)) + eps);  % Normalize
    
    % --- 3. Apply ANC if enabled ---
    if anc_enabled
        for ch = 1:numChannels
            for n = filterOrder:frameSize
                % Use mono micData for both stereo channels
                x_ref = micData(n:-1:n-filterOrder+1);  
                y_hat = w(:,ch).' * x_ref;
                chunk(n,ch) = chunk(n,ch) - y_hat;
                w(:,ch) = w(:,ch) + mu * chunk(n,ch) * x_ref;
            end
        end
    end
    
    % --- 4. Play audio ---
    speakerWriter(chunk);
    
    % --- 5. Update status text ---
    if anc_enabled
        set(statusText,'String','ANC: ON');
    else
        set(statusText,'String','ANC: OFF');
    end
    
    drawnow limitrate;
end

disp('ANC terminated.');

%% --- Cleanup ---
release(micReader);
release(speakerWriter);
if ishandle(hFig), close(hFig); end

%% --- GUI callbacks ---
function toggleANC(~,~)
    global anc_enabled
    anc_enabled = ~anc_enabled;
    if anc_enabled
        disp('Noise Cancelling: ON');
    else
        disp('Noise Cancelling: OFF');
    end
end

function quitANC(~,~)
    global running
    running = false;
end

function quitFcn(~,~)
    global running
    running = false;
    delete(gcf);
end