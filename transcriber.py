import os
import re
import json
import shutil
import warnings
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Program to transcribe audio files.")
parser.add_argument('file', type=str, help="The target audio file.")
parser.add_argument('--token', type=str, default=' ', help="Huggingface user token. (Must be set during the first time to download models)")
parser.add_argument('--model', type=str, default='medium', help="Model size (e.g., 'small', 'medium', 'large'). Default: medium")
parser.add_argument('--lang', type=str, default="ja", help="Language of the input data (e.g., 'ja', 'en'). Default: ja")
parser.add_argument('--translate', type=str, help="Target language for translation (e.g., 'ja', 'fr').")

args = parser.parse_args()

filename = os.path.basename(args.file)
lang = args.lang
translate = args.translate if args.translate else None
whisper_size = args.model
source_path = os.path.dirname(args.file)
access_token = args.token


import torch
import whisper
from pydub import AudioSegment
from googletrans import Translator
from pyannote.audio import Pipeline


warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
translator = Translator()

def timeStr(t):
    return '{0:02d}:{1:02d}:{2:06.2f}'.format(round(t // 3600),
                                                round(t % 3600 // 60),
                                                t % 60)

def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
    return s

try:
    temp_path = "temp"
    source_file = os.path.join(source_path, filename)
    dest_path = os.path.join(temp_path, "processed", filename)
    Path(os.path.join(temp_path, "processed")).mkdir(parents=True, exist_ok=True)
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = AudioSegment.from_wav(source_file)
    audio = spacer.append(audio, crossfade=0)
    audio.export(dest_path, format='wav')

    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=access_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print("Great, Cuda is available.") if torch.cuda.is_available() else print("No Cuda available, use CPU instead. (Very Slow)")
    print("Starting speaker diarization task.")
    audio_file = {'uri': 'blabla', 'audio': dest_path}
    dz = pipeline(audio_file)

    with open(os.path.join(temp_path, "diarization.txt"), "w") as text_file:
        text_file.write(str(dz))

    dzs = open(os.path.join(temp_path, "diarization.txt")).read().splitlines()
    print("Diarization task is complete.")
    groups = []
    g = []
    lastend = 0

    for d in dzs:
        if g and (g[0].split()[-1] != d.split()[-1]):
            groups.append(g)
            g = []
        g.append(d)
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
        end = millisec(end)
        if (lastend > end):
            groups.append(g)
            g = []
        else:
            lastend = end
    if g:
        groups.append(g)

    Path(temp_path).mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_wav(dest_path)
    gidx = -1
    for g in groups:
        start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
        start = millisec(start)
        end = millisec(end)
        gidx += 1
        audio[start:end].export(os.path.join(temp_path, str(gidx) + '.wav'), format='wav')

    del audio_file, pipeline, spacer, audio, dz

    print("Starting transcription task.")
    whisper_model = whisper.load_model('medium', device=device)

    for i in range(len(groups)):
        audiof = os.path.join(temp_path, str(i) + '.wav')
        result = whisper_model.transcribe(audio=audiof, language=lang, word_timestamps=True)
        with open(os.path.join(temp_path, str(i)+'.json'), "w") as outfile:
            json.dump(result, outfile, indent=4)
    print("Transcription task is complete.")
    def_boxclr = 'white'
    def_spkrclr = 'orange'
    audio_title = filename
    preS = '\n<!DOCTYPE html>\n<html lang="en">\n\n<head>\n\t<meta charset="UTF-8">\n\t<meta name="viewport" content="whtmlidth=device-width, initial-scale=1.0">\n\t<meta http-equiv="X-UA-Compatible" content="ie=edge">\n\t<title>' + \
    audio_title+ \
    '</title>\n\t<style>\n\t\tbody {\n\t\t\tfont-family: sans-serif;\n\t\t\tfont-size: 14px;\n\t\t\tcolor: #111;\n\t\t\tpadding: 0 0 1em 0;\n\t\t\tbackground-color: #efe7dd;\n\t\t}\n\n\t\ttable {\n\t\t\tborder-spacing: 10px;\n\t\t}\n\n\t\tth {\n\t\t\ttext-align: left;\n\t\t}\n\n\t\t.lt {\n\t\t\tcolor: inherit;\n\t\t\ttext-decoration: inherit;\n\t\t}\n\n\t\t.l {\n\t\t\tcolor: #050;\n\t\t}\n\n\t\t.s {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.c {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.e {\n\t\t\t/*background-color: white; Changing background color */\n\t\t\tborder-radius: 10px;\n\t\t\t/* Making border radius */\n\t\t\twidth: 50%;\n\t\t\t/* Making auto-sizable width */\n\t\t\tpadding: 0 0 0 0;\n\t\t\t/* Making space around letters */\n\t\t\tfont-size: 14px;\n\t\t\t/* Changing font size */\n\t\t\tmargin-bottom: 0;\n\t\t}\n\n\t\t.t {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t#player-div {\n\t\t\tposition: sticky;\n\t\t\ttop: 20px;\n\t\t\tfloat: right;\n\t\t\twidth: 40%\n\t\t}\n\n\t\t#player {\n\t\t\taspect-ratio: 16 / 9;\n\t\t\twidth: 100%;\n\t\t\theight: auto;\n\t\t}\n\n\t\ta {\n\t\t\tdisplay: inline;\n\t\t}\n\t</style>';
    preS += '\n\t<script>\n\twindow.onload = function () {\n\t\t\tvar player = document.getElementById("audio_player");\n\t\t\tvar player;\n\t\t\tvar lastword = null;\n\n\t\t\t// So we can compare against new updates.\n\t\t\tvar lastTimeUpdate = "-1";\n\n\t\t\tsetInterval(function () {\n\t\t\t\t// currentTime is checked very frequently (1 millisecond),\n\t\t\t\t// but we only care about whole second changes.\n\t\t\t\tvar ts = (player.currentTime).toFixed(1).toString();\n\t\t\t\tts = (Math.round((player.currentTime) * 5) / 5).toFixed(1);\n\t\t\t\tts = ts.toString();\n\t\t\t\tconsole.log(ts);\n\t\t\t\tif (ts !== lastTimeUpdate) {\n\t\t\t\t\tlastTimeUpdate = ts;\n\n\t\t\t\t\t// Its now up to you to format the time.\n\t\t\t\t\tword = document.getElementById(ts)\n\t\t\t\t\tif (word) {\n\t\t\t\t\t\tif (lastword) {\n\t\t\t\t\t\t\tlastword.style.fontWeight = "normal";\n\t\t\t\t\t\t}\n\t\t\t\t\t\tlastword = word;\n\t\t\t\t\t\t//word.style.textDecoration = "underline";\n\t\t\t\t\t\tword.style.fontWeight = "bold";\n\n\t\t\t\t\t\tlet toggle = document.getElementById("autoscroll");\n\t\t\t\t\t\tif (toggle.checked) {\n\t\t\t\t\t\t\tlet position = word.offsetTop - 20;\n\t\t\t\t\t\t\twindow.scrollTo({\n\t\t\t\t\t\t\t\ttop: position,\n\t\t\t\t\t\t\t\tbehavior: "smooth"\n\t\t\t\t\t\t\t});\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}, 0.1);\n\t\t}\n\n\t\tfunction jumptoTime(timepoint, id) {\n\t\t\tvar player = document.getElementById("audio_player");\n\t\t\thistory.pushState(null, null, "#" + id);\n\t\t\tplayer.pause();\n\t\t\tplayer.currentTime = timepoint;\n\t\t\tplayer.play();\n\t\t}\n\t\t</script>\n\t</head>';
    preS += '\n\n<body>\n\t<h2>' + audio_title + '</h2>\n\t<i>Click on a part of the transcription, to jump to its portion of audio, and get an anchor to it in the address\n\t\tbar<br><br></i>\n\t<div id="player-div">\n\t\t<div id="player">\n\t\t\t<audio controls="controls" id="audio_player">\n\t\t\t\t<source src="'+ source_file +'" />\n\t\t\t</audio>\n\t\t</div>\n\t\t<div><label for="autoscroll">auto-scroll: </label>\n\t\t\t<input type="checkbox" id="autoscroll" checked>\n\t\t</div>\n\t</div>\n';
    postS = '\t</body>\n</html>'

    html = list(preS)
    txt = list("")
    gidx = -1

    if translate:
        print("Starting translation task.")
        lines = []
        for idx, g in enumerate(groups):
            lines.append("".join([text["text"] for text in json.load(open(os.path.join(temp_path, str(idx) + '.json')))['segments']]))
        lines = [string if string else " " for string in lines]
        batch_results = translator.translate(lines, dest=translate)
        translations = [result.text for result in batch_results]
        print("Translation task is complete.")

    for g in groups:
        shift = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        shift = millisec(shift) - spacermilli
        shift=max(shift, 0)

        gidx += 1

        captions = json.load(open(os.path.join(temp_path, str(gidx) + '.json')))['segments']

        if captions:
            speaker = g[0].split()[-1]
            boxclr = def_boxclr
            spkrclr = def_spkrclr

            html.append(f'<div class="e" style="background-color: {boxclr}">\n');
            html.append('<p  style="margin:0;padding: 5px 10px 10px 10px;word-wrap:normal;white-space:normal;">\n')
            html.append(f'<span style="color:{spkrclr};font-weight: bold;">{speaker}</span><br>\n\t\t\t\t')

        for c in captions:
            start = shift + c['start'] * 1000.0
            start = start / 1000.0
            end = (shift + c['end'] * 1000.0) / 1000.0
            txt.append(f'[{timeStr(start)} --> {timeStr(end)}] [{speaker}] {c["text"]}\n')
            for i, w in enumerate(c['words']):
                if w == "":
                    continue
                start = (shift + w['start']*1000.0) / 1000.0
                html.append(f'<a href="#{timeStr(start)}" id="{"{:.1f}".format(round(start*5)/5)}" class="lt" onclick="jumptoTime({int(start)}, this.id)">{w["word"]}</a><!--\n\t\t\t\t-->')
        html.append('</p>\n')
        if translate:
            html.append('<p  style="margin:0;padding: 5px 10px 10px 10px;word-wrap:normal;white-space:normal;">\n')
            html.append(translations[gidx])
            html.append('</p>\n')
        html.append(f'</div>\n')

    html.append(postS)
    with open(f"{filename}.html", "w", encoding='utf-8') as file:
        html_str = "".join(html)
        file.write(html_str)

finally:
    shutil.rmtree(temp_path)
    print("The cleanup process has been completed.")
