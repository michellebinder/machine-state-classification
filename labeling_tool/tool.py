import matplotlib
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import SpanSelector, Button
import pandas as pd
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import messagebox
import time
import matplotlib.dates as mdates
import os


root = tk.Tk()
root.withdraw()

last_click_time = 0
last_clicked_rect = None

def preprocess_dataframe(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if '_time' in df.columns:
        df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
        df['_time'] = df['_time'].dt.tz_localize(None)
    for col in df.columns:
        if col != '_time':
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = [c for c in df.columns if c != '_time' and pd.api.types.is_numeric_dtype(df[c])]
    cols_all_zero = [c for c in numeric_cols if (df[c].fillna(0) == 0).all()]
    df = df.drop(columns=cols_all_zero, errors='ignore')

    numeric_cols = [c for c in df.columns if c != '_time' and pd.api.types.is_numeric_dtype(df[c])]
    const_cols = [c for c in numeric_cols if df[c].nunique() == 1]
    df = df.drop(columns=const_cols, errors='ignore')

    if '_time' in df.columns:
        df = df.sort_values('_time')
    return df

def darken_color(color, factor=0.7):
    c = mcolors.to_rgb(color)
    return (c[0]*factor, c[1]*factor, c[2]*factor)

csv_file = 'Kombi_2024_11_12_german.csv'
original_df = pd.read_csv(csv_file, sep=";")
original_df['Label'] = ''
original_df['_time'] = pd.to_datetime(original_df['_time'], errors='coerce')
original_df['_time'] = original_df['_time'].dt.tz_localize(None)

df = preprocess_dataframe(original_df.copy())
numeric_cols = [col for col in df.columns if col != '_time' and pd.api.types.is_numeric_dtype(df[col])]
if not numeric_cols:
    raise ValueError("No numeric columns found after preprocessing.")

initial_states = {c: (c == 'AggHoeheIst') for c in numeric_cols}

root.update_idletasks()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# DPI abrufen (Standard ist 100, kann aber angepasst werden)
dpi = 100

# Berechne die figsize in Zoll
fig_width = screen_width / dpi
fig_height = (screen_height - 65) / dpi 

fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

cmap = plt.get_cmap('tab20', len(numeric_cols))
lines = []
for i, col in enumerate(numeric_cols):
    color = cmap(i)
    line, = ax.plot(df['_time'], df[col], linestyle='-', label=col, visible=initial_states[col], color=color)
    lines.append(line)

ax.set_title("Labeling Tool", fontsize=14)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Value", fontsize=12)

def update_legend():
    visible_lines = [l for l in lines if l.get_visible()]
    visible_labels = [l.get_label() for l in visible_lines]
    ax.legend(visible_lines, visible_labels, loc='upper right', fontsize=10)

update_legend()

data_min = mdates.date2num(df['_time'].min())
data_max = mdates.date2num(df['_time'].max())

# Alle 15 Minuten für die Ticks
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))  # 15-Minuten-Ticks

# Stündliche Beschriftung
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Stündliche Hauptticks
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format für stündliche Beschriftung

ax.set_xlim(df['_time'].min(), df['_time'].max())

ax.tick_params(axis='x', which='both', rotation=45)
ax.tick_params(axis='y', which='both', left=False)

current_state = None
selected_areas = []
delete_mode = False
dragging = False
dragged_rect = None
click_x = 0
rect_original_x = 0
rect_original_width = 0

state_colors = {
    "Block einladen": "#AEC6CF",
    "Seitenbesäumung": "#77DD77",
    "Schopfbesäumung": "#FFB347",
    "Kleben": "#CDA4DE",
    "Bodenhaut entfernen": "#FF6961", 
    "Produktion": "#64FF33", 
    "Stillstand": "#3633FF",
}

def update_ylim():
    visible_lines = [l for l in lines if l.get_visible()]
    if visible_lines and any(len(l.get_ydata()) > 0 for l in visible_lines):
        all_data = np.concatenate([l.get_ydata()[~np.isnan(l.get_ydata())] for l in visible_lines])
        if len(all_data) > 0:
            ymin, ymax = np.min(all_data), np.max(all_data)
            margin = (ymax - ymin) * 0.1
            ax.set_ylim(ymin - margin, ymax + margin)
        else:
            ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, 1)

def x_to_time(x):
    return mdates.num2date(x).replace(tzinfo=None)

def get_areas_sorted():
    return sorted(selected_areas, key=lambda r: r[0].get_x())

def no_overlap(x_start, x_end, exclude_rect):
    areas = get_areas_sorted()
    for (r2, st2) in areas:
        if r2 is exclude_rect:
            continue
        x0 = r2.get_x()
        w2 = r2.get_width()
        x1 = x0 + w2
        if x_start < x1 and x_end > x0:
            return False
    return True

def clamp_no_overlap(x_start, x_end, exclude_rect, original_x_start=None):
    width = x_end - x_start
    # Clampe an data_min/data_max
    if x_start < data_min:
        x_start = data_min
        x_end = x_start + width
    if x_end > data_max:
        x_end = data_max
        x_start = x_end - width

    # Prüfe ob overlapfrei
    if no_overlap(x_start, x_end, exclude_rect):
        return (x_start, x_end)
    return None

def rebuild_labels():
    original_df['Label'] = ''
    for (r, st) in selected_areas:
        x0 = r.get_x()
        w = r.get_width()
        start_point = x_to_time(x0)
        end_point = x_to_time(x0 + w)
        mask = (original_df['_time'] >= start_point) & (original_df['_time'] <= end_point)
        original_df.loc[mask, 'Label'] = st

def check_all_labeled():
    return not original_df['Label'].isnull().any() and not (original_df['Label'] == '').any()

def in_existing_area(x):
    for (r, st) in reversed(selected_areas):
        x0 = r.get_x()
        w = r.get_width()
        if x0 <= x <= x0 + w:
            return r, st
    return None, None

def remove_area(r):
    r.remove()
    for i, (rr, st) in enumerate(selected_areas):
        if rr is r:
            selected_areas.pop(i)
            break
    rebuild_labels()
    fig.canvas.draw_idle()

def onselect(xmin, xmax):
    global current_state
    if current_state is None:
        messagebox.showwarning("No State", "Please select a state first.")
        return
    
# Clampe direkt am Datenrand ohne weitere Logik:
    width = xmax - xmin
    if xmin < data_min:
        xmin = data_min
        xmax = xmin + width
    if xmax > data_max:
        xmax = data_max
        xmin = xmax - width

    pos = clamp_no_overlap(xmin, xmax, None, None)
    if pos is None:
         # Overlap beim Erstellen: versuche die Markierung auf die zulässige Grenze zu kürzen
        width = xmax - xmin
        # Finde ersten Overlap
        areas = get_areas_sorted()
        overlapped = False
        for (r2, st2) in areas:
            x0 = r2.get_x()
            w2 = r2.get_width()
            x1 = x0 + w2
            # Check Overlap:
            if xmin < x1 and xmax > x0:
                overlapped = True
                x_start = xmin
                x_end = xmax

                overlap_left = (xmax > x0 and xmin < x0)
                overlap_right = (xmin < x1 and xmax > x1)

                if overlap_left and not overlap_right:
                    # Kürze so, dass x_end = x0
                    x_end = x0
                elif overlap_right and not overlap_left:
                    # Kürze so, dass x_start = x1
                    x_start = x1
                else:
                    # Beidseitiger Overlap? Nimm die nächstliegende Grenze
                    dist_to_x0 = abs(xmax - x0)
                    dist_to_x1 = abs(xmin - x1)
                    if dist_to_x0 < dist_to_x1:
                        x_end = x0
                    else:
                        x_start = x1

                # Prüfe Bounds
                new_width = x_end - x_start
                if new_width <= 0:
                    # Kein Platz
                    print("No space after trimming.")
                    return
                if x_start < data_min:
                    diff = data_min - x_start
                    x_start = data_min
                    x_end = x_start + new_width
                if x_end > data_max:
                    diff = x_end - data_max
                    x_end = data_max
                    x_start = x_end - new_width

                new_pos = clamp_no_overlap(x_start, x_end, None, None)
                if new_pos is None:
                    print("Even after trimming, no space.")
                    return
                x_st, x_en = new_pos
                if x_en <= x_st:
                    print("No space after trimming and checking again.")
                    return
                new_area = Rectangle(
                    (x_st, ax.get_ylim()[0]),
                    x_en - x_st,
                    ax.get_ylim()[1] - ax.get_ylim()[0],
                    color=state_colors.get(current_state, 'red'), alpha=0.3
                )
                ax.add_patch(new_area)
                selected_areas.append((new_area, current_state))
                fig.canvas.draw_idle()
                rebuild_labels()
                return
        if not overlapped:
            print("No space to create a new area, even after attempting trimming.")
            return
    else:
        x_start, x_end = pos
        if x_end <= x_start:
            print("No space after clamp.")
            return
        new_area = Rectangle(
            (x_start, ax.get_ylim()[0]),
            x_end - x_start,
            ax.get_ylim()[1] - ax.get_ylim()[0],
            color=state_colors.get(current_state, 'red'), alpha=0.3
        )
        ax.add_patch(new_area)
        selected_areas.append((new_area, current_state))
        fig.canvas.draw_idle()
        rebuild_labels()

        # Prüfe, ob neben dem neu erstellten Bereich ein anderer Bereich so nah ist,
        # dass der Abstand < 0.0001 ist. Finde benachbarte Bereiche:
        threshold = 0.0001
        areas_sorted = get_areas_sorted()
        # Finde den Index unseres neuen Bereichs
        idx = None
        for i,(r,st) in enumerate(areas_sorted):
            if r is new_area:
                idx = i
                break
        if idx is not None:
            # Prüfe linken Nachbar
            if idx > 0:
                left_r, left_st = areas_sorted[idx-1]
                left_x0 = left_r.get_x()
                left_w = left_r.get_width()
                left_x1 = left_x0 + left_w
                # Abstand zwischen left_x1 und x_start
                if abs(x_start - left_x1) < threshold:
                    # Lücke schließen
                    new_area.set_x(left_x1)
                    new_area.set_width(x_end - left_x1)
            # Prüfe rechten Nachbar
            if idx < len(areas_sorted)-1:
                right_r, right_st = areas_sorted[idx+1]
                right_x0 = right_r.get_x()
                # Abstand zwischen x_end und right_x0
                x_start_new = new_area.get_x()
                x_end_new = x_start_new + new_area.get_width()
                if abs(right_x0 - x_end_new) < threshold:
                    # Lücke schließen
                    new_width = (right_x0 - x_start_new)
                    if new_width > 0:
                        new_area.set_width(new_width)

            # Nach dem Anpassen der Grenzen nochmals Labels neu aufbauen
            rebuild_labels()
            fig.canvas.draw_idle()

span = SpanSelector(ax, onselect, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor='red'))

def on_button_click_state(event, state):
    global current_state
    for s, info in state_button_objects.items():
        info['button'].ax.set_facecolor(state_colors[s])
        info['button'].color = state_colors[s]
        info['button'].hovercolor = state_colors[s]

    current_state = state
    dark_col = darken_color(state_colors[state])
    state_button_objects[state]['button'].ax.set_facecolor(dark_col)
    state_button_objects[state]['button'].color = dark_col
    state_button_objects[state]['button'].hovercolor = dark_col
    fig.canvas.draw_idle()

    if not dragging and not delete_mode:
        span.active = True
        span.rect.set_facecolor(state_colors[state])

button_width = 0.12
button_height = 0.05
y_start_right = 0.85
state_buttons = ["Block einladen", "Seitenbesäumung", "Schopfbesäumung", "Kleben", "Bodenhaut entfernen", "Produktion", "Stillstand"]
state_button_objects = {}
for i, sb in enumerate(state_buttons):
    y_btn = y_start_right - i*(button_height+0.01)
    ax_b = plt.axes([0.84, y_btn, button_width, button_height])
    b = Button(ax_b, sb, color=state_colors[sb], hovercolor=state_colors[sb])
    b.on_clicked(lambda event, s=sb: on_button_click_state(event, s))
    state_button_objects[sb] = {'button': b}

# 1. Callback-Funktionen definieren
def zoom_in(event):
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    x_center = (x_max + x_min) / 2
    zoom_factor = 0.9
    new_range = x_range * zoom_factor
    new_x_min = x_center - new_range / 2
    new_x_max = x_center + new_range / 2
    ax.set_xlim(new_x_min, new_x_max)
    fig.canvas.draw_idle()

def zoom_out(event):
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    x_center = (x_max + x_min) / 2
    zoom_factor = 1.1
    new_range = x_range * zoom_factor
    new_x_min = x_center - new_range / 2
    new_x_max = x_center + new_range / 2
    ax.set_xlim(new_x_min, new_x_max)
    fig.canvas.draw_idle()

# 2. Zoom-Buttons erstellen
# Definiere die Breite und Höhe der Zoom-Buttons
zoom_button_width = 0.12
zoom_button_height = 0.05

# Positioniere den Zoom-In Button
ax_zoom_in = plt.axes([0.84, 0.38, zoom_button_width, zoom_button_height])
zoom_in_button = Button(ax_zoom_in, "+ Zoom In", color='lightgray', hovercolor='gray')
zoom_in_button.on_clicked(zoom_in)

# Positioniere den Zoom-Out Button
ax_zoom_out = plt.axes([0.84, 0.32, zoom_button_width, zoom_button_height])
zoom_out_button = Button(ax_zoom_out, "- Zoom Out", color='lightgray', hovercolor='gray')
zoom_out_button.on_clicked(zoom_out)

plt.subplots_adjust(left=0.25, right=0.82, top=0.9, bottom=0.15)
# Toolbar sicherstellen
manager = plt.get_current_fig_manager()
if hasattr(manager, 'toolbar'):
    manager.toolbar.update()
button_height_single = 0.03
button_spacing_left = 0.005
y_start_left = 0.87
button_objects = {}

def toggle_line_visibility(label):
    index = numeric_cols.index(label)
    line = lines[index]
    line.set_visible(not line.get_visible())
    new_bg = '0.7' if line.get_visible() else '0.9'
    button_objects[label].ax.set_facecolor(new_bg)
    button_objects[label].color = new_bg
    button_objects[label].hovercolor = new_bg
    fig.canvas.draw_idle()
    update_ylim()
    update_legend()
    plt.draw()

for col in numeric_cols:
    ax_btn = plt.axes([0.02, y_start_left, 0.16, button_height_single])
    bg_color = '0.7' if initial_states[col] else '0.9'
    btn = Button(ax_btn, col, color=bg_color, hovercolor=bg_color)
    button_objects[col] = btn
    btn.on_clicked(lambda event, c=col: toggle_line_visibility(c))
    y_start_left -= (button_height_single + button_spacing_left)

reset_button_height = 0.04
reset_button_y = y_start_left - 0.02
ax_reset = plt.axes([0.02, reset_button_y, 0.16, reset_button_height])
reset_button = Button(ax_reset, "Reset columns\nto AggHoeheIst", color='lightcoral', hovercolor='salmon')

def reset_to_agg_hoehe_ist(event):
    for i, col in enumerate(numeric_cols):
        lines[i].set_visible(False)
        button_objects[col].ax.set_facecolor('0.9')
        button_objects[col].color = '0.9'
        button_objects[col].hovercolor = '0.9'
    if 'AggHoeheIst' in numeric_cols:
        idx = numeric_cols.index('AggHoeheIst')
        lines[idx].set_visible(True)
        button_objects['AggHoeheIst'].ax.set_facecolor('0.7')
        button_objects['AggHoeheIst'].color = '0.7'
        button_objects['AggHoeheIst'].hovercolor = '0.7'
    fig.canvas.draw_idle()
    update_ylim()
    update_legend()
    plt.draw()

reset_button.on_clicked(reset_to_agg_hoehe_ist)

delete_button_ax = plt.axes([0.84, 0.25, 0.12, 0.05])
delete_button = Button(delete_button_ax, "Delete Mode: OFF", color='lightgray', hovercolor='gray')

def toggle_delete_mode(event):
    global delete_mode
    delete_mode = not delete_mode
    if delete_mode:
        delete_button.ax.set_facecolor(darken_color('lightgray',0.7))
        delete_button.label.set_text("Delete Mode: ON")
        span.active = False
    else:
        delete_button.ax.set_facecolor('lightgray')
        delete_button.label.set_text("Delete Mode: OFF")
        if current_state is not None and not dragging:
            span.active = True
    fig.canvas.draw_idle()

delete_button.on_clicked(toggle_delete_mode)

ax_export = plt.axes([0.84, 0.15, 0.12, 0.05])
export_button = Button(ax_export, "Export CSV", color='lightgray', hovercolor='gray')

def export_csv(event):
    if not check_all_labeled():
        # Fehlende Labels mit '0' auffüllen
        original_df['Label'] = original_df['Label'].replace('', '0')
        messagebox.showwarning(
            "Missing Labels",
            "Some datapoints have not been labeled. These will be exported with '0' as the label."
        )
    # Exportieren der CSV
    # Dateinamen anpassen
    base_name, ext = os.path.splitext(csv_file)
    export_file_name = f"{base_name}_labeled{ext}"
    original_df.to_csv(export_file_name, sep=';', index=False)
    messagebox.showinfo("Export", "Export successful!")

export_button.on_clicked(export_csv)

def on_mouse_press(event):
    global dragging, dragged_rect, click_x, rect_original_x, rect_original_width
    if event.inaxes != ax:
        return
    if event.button == 1:
        r, st = in_existing_area(event.xdata)
        if event.dblclick:
            # Doppelklick
            if r is not None:
                # Doppelklick auf bestehenden Bereich
                rebuild_labels()
                fig.canvas.draw_idle()
            else:
                # Doppelklick auf freien Bereich
                if current_state is None:
                    print("please select a state first")
                    return

                # Finde Nachbarn
                x_click = event.xdata
                areas_sorted = get_areas_sorted()
                # Bestimme linken und rechten Nachbar:
                left_bound = data_min
                right_bound = data_max

                # Suche Position, wo wir x_click einfügen würden
                idx = 0
                for i,(rr,stt) in enumerate(areas_sorted):
                    if rr.get_x() + rr.get_width() < x_click:
                        idx = i+1

                if idx > 0:
                    # linker Nachbar vorhanden
                    left_r, left_st = areas_sorted[idx-1]
                    left_x0 = left_r.get_x()
                    left_w = left_r.get_width()
                    left_x1 = left_x0 + left_w
                    left_bound = left_x1

                if idx < len(areas_sorted):
                    # rechter Nachbar vorhanden
                    right_r, right_st = areas_sorted[idx]
                    right_x0 = right_r.get_x()
                    # rechter Nachbar beginnt bei right_x0
                    right_bound = right_x0

                # Erzeuge neuen Bereich zwischen left_bound und right_bound
                if right_bound <= left_bound:
                    # Kein Platz
                    print("No space to create a new area on double-click.")
                    return

                new_area = Rectangle(
                    (left_bound, ax.get_ylim()[0]),
                    right_bound - left_bound,
                    ax.get_ylim()[1] - ax.get_ylim()[0],
                    color=state_colors.get(current_state, 'red'), alpha=0.3
                )
                ax.add_patch(new_area)
                selected_areas.append((new_area, current_state))
                rebuild_labels()
                fig.canvas.draw_idle()
        else:
            # Einzelklick
            if r is not None:
                # Klick auf bestehenden Bereich
                if delete_mode:
                    remove_area(r)
                    span.active = (current_state is not None and not dragging and not delete_mode)
                else:
                    dragging = True
                    dragged_rect = r
                    click_x = event.xdata
                    rect_original_x = r.get_x()
                    rect_original_width = r.get_width()
                    span.active = False
            else:
                # Klick auf freien Bereich
                if delete_mode:
                    span.active = False
                else:
                    if current_state is None:
                        print("please select a state first")
                        span.active = False
                    else:
                        if not dragging:
                            span.active = True

def get_areas_sorted():
    return sorted(selected_areas, key=lambda r: r[0].get_x())


def on_mouse_move(event):
    if event.inaxes != ax:
        return
    if dragging and dragged_rect is not None:
        dx = event.xdata - click_x
        new_x_start = rect_original_x + dx
        x_end = new_x_start + rect_original_width

        pos = clamp_no_overlap(new_x_start, x_end, dragged_rect, rect_original_x)
        if pos is not None:
            x_start_clamped, x_end_clamped = pos
            # Nur aktualisieren, wenn der Bereich gültig ist:
            if x_end_clamped > x_start_clamped:
                dragged_rect.set_x(x_start_clamped)
                dragged_rect.set_width(x_end_clamped - x_start_clamped)
                rebuild_labels()
                fig.canvas.draw_idle()
            else:
                # Kein Verschieben, da ungültig
                pass
        else:
            # Kein overlapfreier Platz -> Bereich nicht weiter verschieben.
            pass

def on_mouse_release(event):
    global dragging, dragged_rect
    if event.button == 1:
        if dragging:
            dragging = False
            dragged_rect = None
        if current_state is not None and not delete_mode:
            span.active = True


fig.canvas.mpl_connect('button_press_event', on_mouse_press)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)

update_ylim()
plt.show(block=True)