# tk_lin_app_simple.py
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sns.set_style("whitegrid")

class LinAppSimple:
    def __init__(self, root):
        self.root = root
        root.title("Mini App Régression Linéaire (simple)")
        root.geometry("1000x650")

        self.data = None
        self.X = None
        self.Y = None
        self.X_enc = None
        self.model = None
        self.coef = None
        self.intercept = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None

        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)

        btn_load = tk.Button(top, text="Charger CSV", command=self.load_csv)
        btn_load.pack(side="left")

        btn_showinfo = tk.Button(top, text="Afficher info & corrélation", command=self.show_info)
        btn_showinfo.pack(side="left", padx=6)

        mid = tk.Frame(root)
        mid.pack(fill="both", expand=True)

        left = tk.Frame(mid, width=320)
        left.pack(side="left", fill="y", padx=6, pady=6)

        tk.Label(left, text="Colonnes (double-clic -> définit Y)").pack(anchor="w")
        self.listbox = tk.Listbox(left, selectmode="extended", width=40, height=14)
        self.listbox.pack(pady=4)
        self.listbox.bind("<Double-Button-1>", self.select_y_from_double)

        tk.Label(left, text="Colonne cible (Y) :").pack(anchor="w", pady=(8,0))
        self.y_var = tk.StringVar()
        self.y_entry = tk.Entry(left, textvariable=self.y_var, width=30)
        self.y_entry.pack()

        tk.Label(left, text="Colonnes explicatives (X) :").pack(anchor="w", pady=(8,0))
        self.x_text = tk.Text(left, height=4, width=38)
        self.x_text.pack()

        btn_select = tk.Button(left, text="Utiliser la sélection (Listbox -> X)", command=self.use_listbox_selection)
        btn_select.pack(fill="x", pady=4)

        btn_train = tk.Button(left, text="Entraîner modèle", command=self.train_model)
        btn_train.pack(fill="x", pady=4)

        tk.Label(left, text="Nombre de bins pour matrice de confusion :").pack(anchor="w", pady=(8,0))
        self.bins_spin = tk.Spinbox(left, from_=2, to=10, width=5)
        self.bins_spin.pack(anchor="w")

        tk.Label(left, text="Coefficients appris :").pack(anchor="w", pady=(8,0))
        self.coef_text = tk.Text(left, height=12, width=38)
        self.coef_text.pack()

        right = tk.Frame(mid)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self.fig, (self.ax_scatter, self.ax_conf) = plt.subplots(1, 2, figsize=(10,4))
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path:
            return
        try:
            self.data = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lire le fichier : {e}")
            return
        self.listbox.delete(0, tk.END)
        for col in self.data.columns:
            self.listbox.insert(tk.END, col)
        messagebox.showinfo("OK", f"Fichier chargé ({len(self.data)} lignes, {len(self.data.columns)} colonnes)")

    def show_info(self):
        if self.data is None:
            messagebox.showwarning("Avertissement", "Charge un CSV d'abord.")
            return
        info_lines = []
        info_lines.append("Aperçu (5 lignes):\n" + self.data.head().to_string())
        info_lines.append("\nColonnes :\n" + ", ".join(self.data.columns))
        na = self.data.isna().sum()
        info_lines.append("\nValeurs manquantes par colonne :\n" + (na[na>0].to_string() if na.sum()>0 else "Aucune valeur manquante."))
        txt = "\n\n".join(info_lines)

        top = tk.Toplevel(self.root)
        top.title("Info dataset")
        t = tk.Text(top, width=110, height=18)
        t.pack()
        t.insert("1.0", txt)

        corr = self.data.corr()
        fig2, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
        ax.set_title("Matrice de corrélation")
        canvas2 = FigureCanvasTkAgg(fig2, master=top)
        canvas2.get_tk_widget().pack()
        canvas2.draw()

    def select_y_from_double(self, event):
        sel = self.listbox.curselection()
        if sel:
            val = self.listbox.get(sel[0])
            self.y_var.set(val)

    def use_listbox_selection(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Avertissement", "Sélectionne d'abord des colonnes dans la liste.")
            return
        cols = [self.listbox.get(i) for i in sel]
        self.x_text.delete("1.0", tk.END)
        self.x_text.insert(tk.END, ", ".join(cols))

    def train_model(self):
        if self.data is None:
            messagebox.showwarning("Avertissement", "Charge un CSV d'abord.")
            return

        y_col = self.y_var.get().strip()
        if y_col == "":
            messagebox.showwarning("Avertissement", "Indique la colonne cible (Y).")
            return
        if y_col not in self.data.columns:
            messagebox.showerror("Erreur", f"Colonne Y '{y_col}' introuvable.")
            return

        x_input = self.x_text.get("1.0", tk.END).strip()
        if x_input.lower() == "all" or x_input == "":
            x_cols = [c for c in self.data.columns if c != y_col]
        else:
            x_cols = [c.strip() for c in x_input.split(",") if c.strip() != ""]

        for c in x_cols:
            if c not in self.data.columns:
                messagebox.showerror("Erreur", f"Colonne X '{c}' introuvable.")
                return

        X = self.data.loc[:, x_cols].copy()
        Y = self.data.loc[:, y_col].copy()

        if X.isna().sum().sum() > 0 or Y.isna().sum() > 0:
            resp = messagebox.askyesno("Valeurs manquantes", "Le dataset contient des valeurs manquantes. Les lignes avec NaN seront supprimées. Continuer ?")
            if not resp:
                return
            df = pd.concat([X, Y], axis=1).dropna()
            X = df.loc[:, x_cols]
            Y = df.loc[:, y_col]

        X_enc = pd.get_dummies(X, drop_first=True)
        self.X = X
        self.Y = Y
        self.X_enc = X_enc

        X_train, X_test, y_train, y_test = train_test_split(X_enc, Y, test_size=0.3, random_state=0)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.model = model
        self.coef = model.coef_.copy()
        self.intercept = float(model.intercept_)
        self.x_test = X_test.reset_index(drop=True)
        self.y_test = pd.Series(y_test).reset_index(drop=True)
        self.y_pred = pd.Series(y_pred).reset_index(drop=True)

        self.coef_text.delete("1.0", tk.END)
        for name, c in zip(X_enc.columns, self.coef):
            self.coef_text.insert(tk.END, f"{name} : {c:.4f}\n")
        self.coef_text.insert(tk.END, f"\nIntercept : {self.intercept:.4f}\n")

        # metrics
        r2 = r2_score(self.y_test, self.y_pred)
        rmse = mean_squared_error(self.y_test, self.y_pred, squared=False)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        metrics_text = f"R²: {r2:.4f}   RMSE: {rmse:.3f}   MAE: {mae:.3f}"
        messagebox.showinfo("Résultats", metrics_text)

        self.update_plots()

    def update_plots(self):
        if self.y_test is None or self.x_test is None:
            return

        self.ax_scatter.clear()
        self.ax_scatter.scatter(self.y_test, self.y_pred, alpha=0.7)
        self.ax_scatter.plot([self.y_test.min(), self.y_test.max()],
                             [self.y_test.min(), self.y_test.max()],
                             color="red", linestyle="--")
        self.ax_scatter.set_title("Réel vs Prédit")
        self.ax_scatter.set_xlabel("y_test")
        self.ax_scatter.set_ylabel("y_pred")

        try:
            n_bins = int(self.bins_spin.get())
        except:
            n_bins = 4

        bins = np.quantile(self.y_test, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)

        self.ax_conf.clear()
        if len(bins) <= 1:
            cm = np.array([[len(self.y_test)]])
            sns.heatmap(cm, annot=True, fmt="d", cmap="plasma", ax=self.ax_conf)
            self.ax_conf.set_title("Matrice de confusion (impossible à construire)")
        else:
            y_test_b = pd.cut(pd.Series(self.y_test), bins=bins, labels=False, include_lowest=True)
            y_pred_b = pd.cut(pd.Series(self.y_pred), bins=bins, labels=False, include_lowest=True)
            y_test_b = pd.Series(y_test_b).fillna(0).astype(int)
            y_pred_b = pd.Series(y_pred_b).fillna(0).astype(int)

            cm = confusion_matrix(y_test_b, y_pred_b)
            sns.heatmap(cm, annot=True, fmt="d", cmap="plasma", ax=self.ax_conf)
            self.ax_conf.set_title("Matrice de confusion (bins)")
            self.ax_conf.set_xlabel("y_pred bins")
            self.ax_conf.set_ylabel("y_true bins")

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = LinAppSimple(root)
    root.mainloop()
