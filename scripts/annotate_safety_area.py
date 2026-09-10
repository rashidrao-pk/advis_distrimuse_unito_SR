#!/usr/bin/env python3
"""Generate a local HTML interface for frame/range safety-area annotation."""

import argparse
from html import escape
import json
from pathlib import Path
import re

import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FRAME_ID_PATTERN = re.compile(r"(?:^|[_-])f(?:rame)?[-_]?([0-9]+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an interactive HTML annotator for one extracted scenario."
    )
    parser.add_argument("scenario_path", type=Path)
    parser.add_argument("--camera", default="back_view")
    parser.add_argument("--areas", nargs="+", help="Include only these safety areas.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/cf_dataset_mac.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML. Default: reports/safety_area_annotations/<scenario>_<camera>.html",
    )
    return parser.parse_args()


def frame_id(path):
    match = FRAME_ID_PATTERN.search(path.stem)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"[0-9]+", path.stem)
    return int(numbers[-1]) if numbers else None


def image_files(directory):
    return sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: (
            frame_id(path) is None,
            frame_id(path) if frame_id(path) is not None else path.name,
        ),
    )


def scenario_description(config, scenario_id):
    option = (config.get("scenario_options", {}) or {}).get(str(scenario_id), {}) or {}
    description = option.get("description")
    if not description:
        selected = config.get("scenario", {}) or {}
        if str(selected.get("id", "")) == str(scenario_id):
            description = selected.get("description")
    return str(description or "Description not available in config.")


def collect_data(scenario_path, camera, selected_areas):
    camera_path = scenario_path / camera
    processed_path = camera_path / "processed"
    if not processed_path.is_dir():
        raise FileNotFoundError(f"Processed safety-area directory not found: {processed_path}")

    area_dirs = sorted(path for path in processed_path.iterdir() if path.is_dir())
    if selected_areas:
        requested = set(selected_areas)
        area_dirs = [path for path in area_dirs if path.name in requested]
        missing = requested - {path.name for path in area_dirs}
        if missing:
            raise ValueError(f"Safety-area folder(s) not found: {', '.join(sorted(missing))}")
    if not area_dirs:
        raise ValueError("No safety-area folders found")

    raw_path = camera_path / "raw"
    raw_by_id = {
        frame_id(path): path.resolve().as_uri()
        for path in image_files(raw_path)
        if frame_id(path) is not None
    } if raw_path.is_dir() else {}

    areas = {}
    for area_dir in area_dirs:
        frames = []
        for index, path in enumerate(image_files(area_dir)):
            identifier = frame_id(path)
            if identifier is None:
                identifier = index
            frames.append({
                "frame_id": identifier,
                "filename": path.name,
                "processed_uri": path.resolve().as_uri(),
                "processed_path": str(path.resolve()),
                "raw_uri": raw_by_id.get(identifier, ""),
                "raw_path": raw_by_id.get(identifier, "").removeprefix("file://"),
            })
        areas[area_dir.name] = frames
    return areas


def build_html(scenario_id, description, camera, areas):
    payload = json.dumps({
        "scenario_id": scenario_id,
        "description": description,
        "camera": camera,
        "areas": areas,
    }).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotate scenario {escape(scenario_id)}</title>
<style>
:root{{--normal:#14804a;--anomaly:#c62828;--verify:#d97706;--accent:#1769aa;--border:#d7dde5}}
*{{box-sizing:border-box}} body{{margin:0;font-family:system-ui,sans-serif;background:#f4f6f8;color:#17212b}}
header{{background:#17212b;color:#fff;padding:14px 22px;position:sticky;top:0;z-index:5}}
header h1{{font-size:1.25rem;margin:0 0 4px}} header p{{margin:0;color:#d7e0e8}}
main{{max-width:1500px;margin:auto;padding:18px}} .panel{{background:#fff;border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:14px}}
.toolbar,.range,.actions{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
button,select,input{{font:inherit;padding:7px 10px}} button{{cursor:pointer;border:1px solid #abb6c3;background:#fff;border-radius:6px}}
button.primary{{background:var(--accent);color:#fff;border-color:var(--accent)}} button.normal{{background:var(--normal);color:#fff}} button.anomaly{{background:var(--anomaly);color:#fff}} button.verify{{background:var(--verify);color:#fff}}
#slider{{width:100%;margin:12px 0}} .counter{{font-weight:650;min-width:190px}}
.viewers{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .viewer{{background:#111;border-radius:8px;overflow:hidden;min-height:320px;display:flex;flex-direction:column}}
.viewer h3{{color:#fff;font-size:.9rem;margin:0;padding:8px 12px;background:#222}} .viewer img{{width:100%;height:min(62vh,650px);object-fit:contain}}
.status{{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px}} .stat{{border:1px solid var(--border);padding:10px;border-radius:7px}}
.stat strong{{display:block;font-size:1.2rem}} #currentLabel.normal{{color:var(--normal)}} #currentLabel.anomalous{{color:var(--anomaly)}} #currentLabel.verify{{color:var(--verify)}}
.timeline-wrap{{position:relative}} #annotationTimeline{{display:block;width:100%;height:54px;border-radius:7px;background:#cbd1d8}}
.timeline-legend{{display:flex;gap:18px;align-items:center;margin-top:8px;color:#59697a;font-size:.88rem}}
.legend-swatch{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}}
textarea{{width:100%;min-height:60px;padding:8px;font:inherit}} .hint{{color:#59697a;font-size:.9rem}} .hidden{{display:none}}
.table-wrap{{max-height:430px;overflow:auto;border:1px solid var(--border);border-radius:7px}}
table{{border-collapse:collapse;width:max-content;min-width:100%;font-size:.86rem}} th,td{{border-bottom:1px solid var(--border);padding:7px 9px;text-align:left;white-space:nowrap}}
th{{position:sticky;top:0;background:#e9eef4;z-index:1}} tr:nth-child(even){{background:#f8fafc}} td.path{{max-width:420px;overflow:hidden;text-overflow:ellipsis}}
@media(max-width:850px){{.viewers{{grid-template-columns:1fr}}.status{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<header><h1>Scenario {escape(scenario_id)} — {escape(description)}</h1><p>Camera: {escape(camera)}</p></header>
<main>
 <section class="panel toolbar">
  <label>Safety area <select id="area"></select></label>
  <span class="counter" id="counter"></span>
  <button id="previous">← Previous</button><button id="next">Next →</button>
  <button id="play">▶ Play</button><label>Speed <select id="speed"><option value="1000">1 fps</option><option value="400" selected>2.5 fps</option><option value="200">5 fps</option><option value="100">10 fps</option><option value="40">25 fps</option></select></label>
  <input id="slider" type="range" min="0" value="0">
 </section>
 <section class="viewers">
  <div class="viewer" id="rawViewer"><h3>Raw frame</h3><img id="rawImage" alt="Raw frame unavailable"></div>
  <div class="viewer"><h3>Safety-area frame</h3><img id="processedImage" alt="Processed safety-area frame"></div>
 </section>
 <section class="panel status">
  <div class="stat">Current label<strong id="currentLabel">Unlabeled</strong></div>
  <div class="stat">Normal<strong id="normalCount">0</strong></div>
  <div class="stat">Anomalous<strong id="anomalyCount">0</strong></div>
  <div class="stat">Verify<strong id="verifyCount">0</strong></div>
  <div class="stat">Unlabeled<strong id="unlabeledCount">0</strong></div>
 </section>
 <section class="panel timeline-wrap">
  <h2>Annotation timeline</h2>
  <canvas id="annotationTimeline" aria-label="Read-only annotation timeline"></canvas>
  <div class="timeline-legend"><span><i class="legend-swatch" style="background:#14804a"></i>Normal</span><span><i class="legend-swatch" style="background:#c62828"></i>Anomalous</span><span><i class="legend-swatch" style="background:#d97706"></i>Verify</span><span><i class="legend-swatch" style="background:#cbd1d8"></i>Unlabeled</span><span>◆ Current frame</span><strong id="timelinePosition"></strong></div>
 </section>
 <section class="panel">
  <h2>Annotate current frame or a sequence</h2>
  <div class="range"><label>From frame position <input id="rangeStart" type="number" min="1"></label><label>To <input id="rangeEnd" type="number" min="1"></label><button id="useCurrentStart">Set start here</button><button id="useCurrentEnd">Set end here</button></div>
  <p class="hint">The range is inclusive. Keep start and end equal to label only the current frame.</p>
  <label>Annotation note<textarea id="note" placeholder="Optional reason or observation"></textarea></label>
  <div class="actions"><button class="normal" data-label="Normal">Label Normal</button><button class="anomaly" data-label="Anomalous">Label Anomalous</button><button class="verify" data-label="Verify">Label Verify</button><button data-label="Unlabeled">Clear label</button></div>
 </section>
 <section class="panel">
  <h2>CSV download preview</h2>
  <p class="hint" id="tableSummary"></p>
  <div class="table-wrap"><table><thead><tr><th>Scenario</th><th>Description</th><th>Camera</th><th>Safety area</th><th>Position</th><th>Frame ID</th><th>Filename</th><th>Label</th><th>Note</th><th>Processed image path</th><th>Raw image path</th></tr></thead><tbody id="csvPreview"></tbody></table></div>
 </section>
 <section class="panel actions">
  <button class="primary" id="download">Download full-scenario CSV</button>
  <label><button id="importButton" type="button">Import annotation CSV</button><input class="hidden" id="importFile" type="file" accept=".csv,text/csv"></label>
  <button id="clearAll">Clear all annotations</button>
  <span class="hint">Labels autosave in this browser. Images stay at their dataset paths and are not stored in this HTML.</span>
 </section>
</main>
<script id="dataset" type="application/json">{payload}</script>
<script>
const data=JSON.parse(document.getElementById('dataset').textContent);
const storageKey=`safety-annotations:${{data.scenario_id}}:${{data.camera}}`;
let annotations=JSON.parse(localStorage.getItem(storageKey)||'{{}}'), index=0, timer=null;
const $=id=>document.getElementById(id), area=$('area'), slider=$('slider');
Object.keys(data.areas).forEach(name=>area.add(new Option(name,name)));
function frames(){{return data.areas[area.value]||[]}} function key(frame){{return `${{area.value}}|${{frame.frame_id}}`}}
function save(){{localStorage.setItem(storageKey,JSON.stringify(annotations))}}
function labelFor(frame){{return (annotations[key(frame)]||{{label:'Unlabeled'}}).label}}
function drawTimeline(){{
 const canvas=$('annotationTimeline'),list=frames();if(!list.length)return;const ratio=window.devicePixelRatio||1,width=Math.max(1,canvas.clientWidth),height=54;
 if(canvas.width!==Math.round(width*ratio)||canvas.height!==Math.round(height*ratio)){{canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);}}
 const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
 const colors={{Normal:'#14804a',Anomalous:'#c62828',Verify:'#d97706',Unlabeled:'#cbd1d8'}},barTop=13,barHeight=27;
 let start=0,current=labelFor(list[0]);for(let i=1;i<=list.length;i++){{const next=i<list.length?labelFor(list[i]):null;if(next!==current){{const x1=start/list.length*width,x2=i/list.length*width;ctx.fillStyle=colors[current]||colors.Unlabeled;ctx.fillRect(x1,barTop,Math.max(1,x2-x1),barHeight);start=i;current=next;}}}}
 const pinX=list.length===1?width/2:index/(list.length-1)*width;ctx.strokeStyle='#111827';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(pinX,5);ctx.lineTo(pinX,46);ctx.stroke();ctx.fillStyle='#111827';ctx.beginPath();ctx.moveTo(pinX,51);ctx.lineTo(pinX-6,42);ctx.lineTo(pinX+6,42);ctx.closePath();ctx.fill();
 $('timelinePosition').textContent=`Current: ${{index+1}} / ${{list.length}}`;
}}
function annotationRows(){{const rows=[];Object.entries(data.areas).forEach(([name,list])=>list.forEach((f,i)=>{{const a=annotations[`${{name}}|${{f.frame_id}}`]||{{label:'Unlabeled',note:''}};rows.push([data.scenario_id,data.description,data.camera,name,i+1,f.frame_id,f.filename,a.label,a.note,f.processed_path,f.raw_path]);}}));return rows;}}
function renderTable(){{const body=$('csvPreview'),fragment=document.createDocumentFragment(),rows=annotationRows();body.textContent='';rows.forEach(row=>{{const tr=document.createElement('tr');row.forEach((value,column)=>{{const td=document.createElement('td');td.textContent=value;td.title=value;if(column>=9)td.className='path';tr.appendChild(td);}});fragment.appendChild(tr);}});body.appendChild(fragment);const labeled=rows.filter(row=>row[7]!=='Unlabeled').length;$('tableSummary').textContent=`${{rows.length}} rows will be downloaded · ${{labeled}} labeled · ${{rows.length-labeled}} unlabeled`;}}
function render(){{
 const list=frames(); if(!list.length)return; index=Math.max(0,Math.min(index,list.length-1)); const frame=list[index];
 slider.max=list.length-1; slider.value=index; $('counter').textContent=`${{area.value}} · ${{index+1}} / ${{list.length}} · frame ${{frame.frame_id}}`;
 $('processedImage').src=frame.processed_uri; $('rawViewer').classList.toggle('hidden',!frame.raw_uri); $('rawImage').src=frame.raw_uri||'';
 $('rangeStart').max=list.length; $('rangeEnd').max=list.length; if(document.activeElement!==$('rangeStart'))$('rangeStart').value=index+1; if(document.activeElement!==$('rangeEnd'))$('rangeEnd').value=index+1;
 const item=annotations[key(frame)]||{{label:'Unlabeled',note:''}}; $('currentLabel').textContent=item.label; $('currentLabel').className=item.label.toLowerCase(); $('note').value=item.note||'';
 let normal=0,anomaly=0,verify=0; list.forEach(f=>{{const l=(annotations[key(f)]||{{}}).label;if(l==='Normal')normal++;if(l==='Anomalous')anomaly++;if(l==='Verify')verify++;}}); $('normalCount').textContent=normal;$('anomalyCount').textContent=anomaly;$('verifyCount').textContent=verify;$('unlabeledCount').textContent=list.length-normal-anomaly-verify;
 drawTimeline();
}}
function move(step){{index=(index+step+frames().length)%frames().length;render()}}
area.onchange=()=>{{index=0;render()}}; slider.oninput=()=>{{index=Number(slider.value);render()}}; $('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);
$('useCurrentStart').onclick=()=>$('rangeStart').value=index+1;$('useCurrentEnd').onclick=()=>$('rangeEnd').value=index+1;
document.querySelectorAll('[data-label]').forEach(button=>button.onclick=()=>{{let start=Number($('rangeStart').value)-1,end=Number($('rangeEnd').value)-1;if(start>end)[start,end]=[end,start];start=Math.max(0,start);end=Math.min(frames().length-1,end);for(let i=start;i<=end;i++){{const k=key(frames()[i]);if(button.dataset.label==='Unlabeled')delete annotations[k];else annotations[k]={{label:button.dataset.label,note:$('note').value.trim()}}}}save();render();renderTable();}});
$('note').onchange=()=>{{const f=frames()[index],old=annotations[key(f)];if(old){{old.note=$('note').value.trim();save();renderTable();}}}};
$('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;$('play').textContent='▶ Play';}}else{{timer=setInterval(()=>move(1),Number($('speed').value));$('play').textContent='⏸ Pause';}}}};
document.onkeydown=e=>{{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(e.key.toLowerCase()==='n')document.querySelector('[data-label="Normal"]').click();if(e.key.toLowerCase()==='a')document.querySelector('[data-label="Anomalous"]').click();if(e.key.toLowerCase()==='v')document.querySelector('[data-label="Verify"]').click();}};
function csvCell(value){{return '"'+String(value??'').replaceAll('"','""')+'"'}}
$('download').onclick=async()=>{{
 const columns=['scenario_id','scenario_description','camera','safety_area','frame_position','frame_id','filename','label','note','processed_image_path','raw_image_path'],rows=[columns,...annotationRows()];
 const blob=new Blob([rows.map(r=>r.map(csvCell).join(',')).join('\\n')],{{type:'text/csv;charset=utf-8'}}),suggestedName=`scenario_${{data.scenario_id}}_${{data.camera}}_annotations.csv`;
 if(window.showSaveFilePicker){{try{{const handle=await window.showSaveFilePicker({{suggestedName,types:[{{description:'CSV annotation file',accept:{{'text/csv':['.csv']}}}}]}});const writable=await handle.createWritable();await writable.write(blob);await writable.close();}}catch(error){{if(error.name!=='AbortError')alert(`Could not save CSV: ${{error.message}}`);}}return;}}
 alert('This browser cannot ask for a save location directly. The CSV will be saved using your browser download settings. Enable “Ask where to save each file” in the browser settings to choose a folder.');
 const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=suggestedName;link.click();URL.revokeObjectURL(link.href);
}};
$('importButton').onclick=()=>$('importFile').click();
$('importFile').onchange=async e=>{{const text=await e.target.files[0].text();const rows=parseCSV(text),head=rows.shift(),col=Object.fromEntries(head.map((x,i)=>[x,i]));rows.forEach(r=>{{const a=r[col.safety_area],id=r[col.frame_id],label=r[col.label];if(a&&id&&['Normal','Anomalous','Verify'].includes(label))annotations[`${{a}}|${{id}}`]={{label,note:r[col.note]||''}};}});save();render();renderTable();}};
function parseCSV(text){{const rows=[];let row=[],cell='',quoted=false;for(let i=0;i<text.length;i++){{const c=text[i];if(c==='"'){{if(quoted&&text[i+1]==='"'){{cell+='"';i++;}}else quoted=!quoted;}}else if(c===','&&!quoted){{row.push(cell);cell='';}}else if((c==='\\n'||c==='\\r')&&!quoted){{if(c==='\\r'&&text[i+1]==='\\n')i++;row.push(cell);if(row.some(x=>x!==''))rows.push(row);row=[];cell='';}}else cell+=c;}}row.push(cell);if(row.some(x=>x!==''))rows.push(row);return rows;}}
$('clearAll').onclick=()=>{{if(confirm('Clear all saved annotations for this scenario and camera?')){{annotations={{}};save();render();renderTable();}}}};new ResizeObserver(drawTimeline).observe($('annotationTimeline'));render();renderTable();
</script></body></html>"""


def main():
    args = parse_args()
    scenario_path = args.scenario_path.expanduser().resolve()
    if not scenario_path.is_dir():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_path}")
    with args.config.expanduser().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    scenario_id = scenario_path.name
    description = scenario_description(config, scenario_id)
    areas = collect_data(scenario_path, args.camera, args.areas)
    output = args.output or (
        Path("reports") / "safety_area_annotations" /
        f"{scenario_id}_{args.camera}_annotator.html"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_html(scenario_id, description, args.camera, areas), encoding="utf-8"
    )
    frame_total = sum(len(frames) for frames in areas.values())
    print(f"Annotator written to: {output}")
    print(f"Safety areas: {', '.join(areas)} | total annotation rows: {frame_total}")
    print("Open the HTML in a browser and use 'Download full-scenario CSV' when done.")


if __name__ == "__main__":
    main()
