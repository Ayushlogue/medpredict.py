import pandas as pd
import numpy as np
import pickle
import os
import sys
import time
import threading
from datetime import datetime


class C:
    RED     = '\033[91m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN    = '\033[96m'
    WHITE   = '\033[97m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    RESET   = '\033[0m'

def red(t):     return f"{C.RED}{C.BOLD}{t}{C.RESET}"
def green(t):   return f"{C.GREEN}{C.BOLD}{t}{C.RESET}"
def yellow(t):  return f"{C.YELLOW}{C.BOLD}{t}{C.RESET}"
def blue(t):    return f"{C.BLUE}{C.BOLD}{t}{C.RESET}"
def magenta(t): return f"{C.MAGENTA}{C.BOLD}{t}{C.RESET}"
def cyan(t):    return f"{C.CYAN}{C.BOLD}{t}{C.RESET}"
def white(t):   return f"{C.WHITE}{C.BOLD}{t}{C.RESET}"
def dim(t):     return f"{C.DIM}{t}{C.RESET}"


class Spinner:
    def __init__(self, message):
        self.message = message
        self.running = False
        self.thread  = None

    def _spin(self):
        frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        i = 0
        while self.running:
            print(f"\r  {cyan(frames[i % len(frames)])}  {self.message}", end="", flush=True)
            time.sleep(0.08)
            i += 1

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._spin)
        self.thread.start()

    def stop(self, done_msg="Done!"):
        self.running = False
        self.thread.join()
        print(f"\r  {green('✔')}  {done_msg}                    ")


def typewrite(text, delay=0.018):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()


def banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║                                                      ║",
        "║        💊  MEDICINE RECOMMENDATION SYSTEM  💊        ║",
        "║             AI / ML  Based  Disease  Predictor       ║",
        "║                                                      ║",
        "╚══════════════════════════════════════════════════════╝",
    ]
    for line in lines:
        print(cyan(line))
        time.sleep(0.05)
    print()
    
    time.sleep(0.3)


def check_files():
    needed = ["model.pkl","symptoms_list.pkl","disease_symptoms.csv","medicines_precautions.csv"]
    missing = [f for f in needed if not os.path.exists(f)]
    if missing:
        print(red(f"\n  ❌ Missing: {', '.join(missing)}"))
        print(yellow("  👉 Run: python train_model.py  first!\n"))
        sys.exit(1)


def load_assets():
    sp = Spinner("Loading AI model...")
    sp.start()
    time.sleep(1.2)
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    with open("symptoms_list.pkl","rb") as f:
        symptoms_list = pickle.load(f)
    med_df = pd.read_csv("medicines_precautions.csv")
    sp.stop("AI Model loaded successfully!")
    time.sleep(0.3)
    return model, symptoms_list, med_df


def section(title):
    pad = 48 - len(title)
    print()
    print(blue(f"  ┌─ {title} {'─' * pad}┐"))

def section_end():
    print(blue(f"  └{'─' * 52}┘"))


def get_user_info():
    section("Patient Information")
    print()
    name = input(white("    👤 Enter your name : ")).strip()
    if not name:
        name = "Patient"
    while True:
        age_input = input(white("    🎂 Enter your age  : ")).strip()
        if age_input.isdigit() and 1 <= int(age_input) <= 120:
            age = int(age_input)
            break
        print(red("    ⚠️  Enter a valid age (1–120)"))
    section_end()
    print()
    typewrite(f"  Hello {green(name)}! 👋  Age: {yellow(str(age))}")
    time.sleep(0.3)
    return name, age


def show_symptoms(symptoms_list):
    symptoms_sorted = sorted(symptoms_list)
    section("Available Symptoms")
    print()
    cols = 3
    items = [(i+1, s.replace("_"," ").title()) for i,s in enumerate(symptoms_sorted)]
    for row_start in range(0, len(items), cols):
        row = items[row_start:row_start+cols]
        line = ""
        for num, name in row:
            entry = f"{dim(str(num).rjust(3)+'.')} {name:<28}"
            line += "  " + entry
        print(line)
    section_end()


def get_symptoms(symptoms_list):
    symptoms_sorted = sorted(symptoms_list)
    selected = []
    section("Enter Your Symptoms")
    print()
    print(dim("    Enter symptom numbers separated by commas."))
    print(dim("    Example: 1, 5, 12, 34"))
    print(dim("    Type 'list' to see symptoms again.\n"))
    while True:
        raw = input(white("    🔢 Symptom numbers: ")).strip()
        if raw.lower() == 'list':
            show_symptoms(symptoms_list)
            continue
        if not raw:
            print(red("    ⚠️  Enter at least one number."))
            continue
        try:
            nums    = [int(x.strip()) for x in raw.split(",")]
            valid   = [n for n in nums if 1 <= n <= len(symptoms_sorted)]
            invalid = [n for n in nums if n < 1 or n > len(symptoms_sorted)]
            if invalid:
                print(yellow(f"    ⚠️  Ignored invalid: {invalid}"))
            if not valid:
                print(red("    ⚠️  No valid numbers. Try again."))
                continue
            selected = [symptoms_sorted[n-1] for n in valid]
            print()
            print(green("    ✅ Symptoms selected:"))
            for s in selected:
                print(cyan(f"       • {s.replace('_',' ').title()}"))
            confirm = input(white("\n    Confirm? (yes/no): ")).strip().lower()
            if confirm in ['yes','y']:
                section_end()
                break
            else:
                print(yellow("    Re-enter your symptoms.\n"))
        except ValueError:
            print(red("    ⚠️  Numbers only, separated by commas."))
    return selected


def predict_disease(model, symptoms_list, selected):
    sp = Spinner("Analyzing symptoms with AI...")
    sp.start()
    time.sleep(1.8)
    symptoms_sorted = sorted(symptoms_list)
    vec = np.array([1 if s in selected else 0 for s in symptoms_sorted]).reshape(1,-1)
    disease = model.predict(vec)[0]
    try:
        probs   = model.predict_proba(vec)[0]
        classes = model.classes_
        top_idx = np.argsort(probs)[::-1][:4]
        top     = [(classes[i], round(probs[i]*100,1)) for i in top_idx if probs[i] > 0]
    except:
        top = [(disease, 100.0)]
    sp.stop("Analysis complete!")
    time.sleep(0.3)
    return disease, top


def get_recommendations(disease, med_df):
    row = med_df[med_df["Disease"].str.strip().str.lower() == disease.strip().lower()]
    if row.empty:
        return [], []
    row  = row.iloc[0]
    meds = [row[c] for c in med_df.columns if "Medicine"   in c and pd.notna(row[c]) and str(row[c]).strip()]
    prec = [row[c] for c in med_df.columns if "Precaution" in c and pd.notna(row[c]) and str(row[c]).strip()]
    return meds, prec


def confidence_bars(top_diseases):
    print()
    section("AI Confidence Chart")
    print()
    bar_width = 30
    colors = [C.GREEN, C.CYAN, C.YELLOW, C.MAGENTA]
    for i, (disease, prob) in enumerate(top_diseases):
        filled = int((prob / 100) * bar_width)
        empty  = bar_width - filled
        bar    = colors[i % len(colors)] + "█" * filled + C.DIM + "░" * empty + C.RESET
        label  = disease[:28].ljust(28)
        pct    = f"{prob:>5.1f}%"
        marker = "◀ TOP MATCH" if i == 0 else ""
        print(f"    {white(label)}  {bar}  {yellow(pct)}  {green(marker)}")
        time.sleep(0.15)
    section_end()


def print_results(name, age, disease, top_diseases, medicines, precautions):
    print()
    print(cyan("  ╔══════════════════════════════════════════════════════╗"))
    print(cyan("  ║") + white("                  📋  DIAGNOSIS REPORT                 ") + cyan("║"))
    print(cyan("  ╚══════════════════════════════════════════════════════╝"))
    print()
    print(f"    {dim('Patient')}  :  {green(name)}")
    print(f"    {dim('Age')}      :  {yellow(str(age))}")
    print()
    typewrite(f"    🔴 Predicted Disease :  {red(disease)}", delay=0.025)
    confidence_bars(top_diseases)
    section("💊 Recommended Medicines")
    print()
    if medicines:
        for m in medicines:
            print(cyan(f"    💊  {m}"))
            time.sleep(0.1)
    else:
        print(yellow("    Consult a doctor for prescription."))
    section_end()
    section("🛡️  Precautions to Follow")
    print()
    if precautions:
        for i, p in enumerate(precautions, 1):
            print(green(f"    {i}.  {p.capitalize()}"))
            time.sleep(0.1)
    section_end()
    print()
    print(yellow("  ⚠️  This is an AI prediction only."))
    print(yellow("     Please consult a qualified doctor.\n"))


def save_report(name, age, disease, medicines, precautions, selected_symptoms):
    save = input(white("  💾 Save this report to a file? (yes/no): ")).strip().lower()
    if save not in ['yes','y']:
        return
    sp = Spinner("Saving report...")
    sp.start()
    time.sleep(0.8)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"report_{name.replace(' ','_')}_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("     MEDICINE RECOMMENDATION SYSTEM — REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"  Date     : {datetime.now().strftime('%d %B %Y, %I:%M %p')}\n")
        f.write(f"  Patient  : {name}\n")
        f.write(f"  Age      : {age}\n\n")
        f.write(f"  Symptoms Entered:\n")
        for s in selected_symptoms:
            f.write(f"    • {s.replace('_',' ').title()}\n")
        f.write(f"\n  Predicted Disease : {disease}\n\n")
        f.write(f"  Recommended Medicines:\n")
        for m in medicines:
            f.write(f"    • {m}\n")
        f.write(f"\n  Precautions:\n")
        for i, p in enumerate(precautions, 1):
            f.write(f"    {i}. {p.capitalize()}\n")
        f.write("\n" + "=" * 55 + "\n")
        f.write("  DISCLAIMER: For educational purposes only.\n")
        f.write("  Always consult a qualified healthcare professional.\n")
        f.write("=" * 55 + "\n")
    sp.stop(f"Report saved as: {filename}")


def ask_again():
    print()
    again = input(white("  🔄 Check another patient? (yes/no): ")).strip().lower()
    return again in ['yes','y']


def main():
    banner()
    check_files()
    model, symptoms_list, med_df = load_assets()
    print()
    show_symptoms(symptoms_list)
    while True:
        name, age    = get_user_info()
        selected     = get_symptoms(symptoms_list)
        disease, top = predict_disease(model, symptoms_list, selected)
        meds, precs  = get_recommendations(disease, med_df)
        print_results(name, age, disease, top, meds, precs)
        save_report(name, age, disease, meds, precs, selected)
        if not ask_again():
            print()
            typewrite(cyan("  Thanks for using Medicine Recommendation System!"))
            typewrite(green("  Stay healthy and take care! 💚"))
            print()
            break

if __name__ == "__main__":
    main()
