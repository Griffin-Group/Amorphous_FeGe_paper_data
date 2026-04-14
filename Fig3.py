import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

# --- 1. Load and Prepare the Data ---
try:
    df = pd.read_csv('J_DMI.txt', sep='\s+')
    print("Successfully loaded J_DMI.txt")
except FileNotFoundError:
    print("Error: 'J_DMI.txt' not found. Please create this file.")
    exit()

try:
    df_sia = pd.read_csv('SIA_eig_max.txt', sep='\s+', skiprows=2, 
                         names=['Type', 'Fe_index', 'eig_max', 'N_Fe', 'N_Ge', 'N_total'])
    print("Successfully loaded SIA_eig_max.txt")
except FileNotFoundError:
    print("Error: 'SIA_eig_max.txt' not found. Please create this file.")
    exit()

df['m_product'] = df['m_i'] * df['m_j']

# --- Helper Functions for Multivariate Polynomial Fitting ---
def multivariate_polyfit(r, mu, z, deg):
    """
    Fits a multivariate polynomial of total degree 'deg':
    z = sum_{i=0}^{deg} sum_{j=0}^{deg-i} c_{ij} * r^i * mu^j
    """
    X = []
    for i in range(deg + 1):
        for j in range(deg - i + 1):
            X.append(r**i * mu**j)
    X = np.stack(X, axis=-1)
    coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    return coeffs

def multivariate_polyval(r, mu, coeffs, deg):
    """
    Evaluates the multivariate polynomial.
    """
    z_fit = np.zeros_like(r, dtype=float)
    idx = 0
    for i in range(deg + 1):
        for j in range(deg - i + 1):
            z_fit += coeffs[idx] * (r**i) * (mu**j)
            idx += 1
    return z_fit

# --- 2. Set Up the Plotting Style ---
plt.rcParams.update({
    'font.size': 16, 
    'axes.titlesize': 30, 
    'axes.labelsize': 24,
    'xtick.labelsize': 16, 
    'ytick.labelsize': 16, 
    'legend.fontsize': 14,
    'figure.titlesize': 18, 
    'lines.linewidth': 1.5,
})

# --- NEW: Custom font size for panel labels ---
panel_fontsize = 28 # Adjust this single value to change all panel labels

# --- 3. Create the Main Grid Layout ---
fig = plt.figure(figsize=(24, 14))
# Define a 5-column grid: plot | plot | cbar | spacer | plot
gs_main = gridspec.GridSpec(2, 5, figure=fig, width_ratios=[20, 20, 1, 0.1, 20])

# Assign axes to the new 5-column grid
ax_a = fig.add_subplot(gs_main[0, 0]) # Col 1
ax_b = fig.add_subplot(gs_main[1, 0]) # Col 1
ax_c = fig.add_subplot(gs_main[0, 1]) # Col 2
ax_d = fig.add_subplot(gs_main[1, 1]) # Col 2
ax_e = fig.add_subplot(gs_main[0, 4]) # Col 5 (index 4)

panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
y_axis_labels = {'J': 'J (meV)', 'D_x': 'D$_x$ (meV)', 'D_y': 'D$_y$ (meV)', 'D_z': 'D$_z$ (meV)'}
norm = TwoSlopeNorm(vmin=df['m_product'].min(), vcenter=0, vmax=df['m_product'].max())

# --- 4. Manually Create Plots ---

# Panel (a)
scatter = ax_a.scatter(df['r_ij'], df['J'], c=df['m_product'], s=120, cmap='bwr', alpha=0.8, edgecolors='black', linewidth=0.5, norm=norm)
ax_a.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
ax_a.set_ylabel(y_axis_labels['J']); ax_a.set_ylim(-600, 100)
ax_a.text(0.02, 0.98, panel_labels[0], transform=ax_a.transAxes, fontsize=panel_fontsize, fontweight='bold', va='top', ha='left')
ax_a.grid(False)
filter_condition = (df['J'] <= -300) | (df['r_ij'] >= 2.5)
df_fit = df[filter_condition]
deg = 3
coeffs = multivariate_polyfit(df_fit['r_ij'], df_fit['m_product'], df_fit['J'], deg)
mu_mean = df_fit['m_product'].mean()
x_fit = np.linspace(df['r_ij'].min(), df['r_ij'].max(), 200)
y_fit = multivariate_polyval(x_fit, np.full_like(x_fit, mu_mean), coeffs, deg)
ax_a.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=2.5, label=f'3rd-order Polynomial Fit')
ax_a.legend(loc='center right')
y_predicted = multivariate_polyval(df_fit['r_ij'], df_fit['m_product'], coeffs, deg)
y_actual = df_fit['J']
ss_res = np.sum((y_actual - y_predicted) ** 2); ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
r2 = 1 - (ss_res / ss_tot)
ax_a.text(0.95, 0.05, f'R² = {r2:.2f}', transform=ax_a.transAxes, fontweight='bold', va='bottom', ha='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))

# Panel (b)
ax_b.scatter(df['r_ij'], df['D_x'], c=df['m_product'], s=120, cmap='bwr', alpha=0.8, edgecolors='black', linewidth=0.5, norm=norm)
ax_b.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
ax_b.set_xlabel('r$_{ij}$ (Å)'); ax_b.set_ylabel(y_axis_labels['D_x'])
ax_b.text(0.02, 0.98, panel_labels[1], transform=ax_b.transAxes, fontsize=panel_fontsize, fontweight='bold', va='top', ha='left')
ax_b.grid(False)

# Panel (c)
ax_c.scatter(df['r_ij'], df['D_y'], c=df['m_product'], s=120, cmap='bwr', alpha=0.8, edgecolors='black', linewidth=0.5, norm=norm)
ax_c.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
ax_c.set_ylabel(y_axis_labels['D_y'])
ax_c.text(0.02, 0.98, panel_labels[2], transform=ax_c.transAxes, fontsize=panel_fontsize, fontweight='bold', va='top', ha='left')
ax_c.grid(False)

# Panel (d)
ax_d.scatter(df['r_ij'], df['D_z'], c=df['m_product'], s=120, cmap='bwr', alpha=0.8, edgecolors='black', linewidth=0.5, norm=norm)
ax_d.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
ax_d.set_xlabel('r$_{ij}$ (Å)'); ax_d.set_ylabel(y_axis_labels['D_z'])
ax_d.text(0.02, 0.98, panel_labels[3], transform=ax_d.transAxes, fontsize=panel_fontsize, fontweight='bold', va='top', ha='left')
ax_d.grid(False)

# Panel (e)
J_fm = df[df['J'] < 0]['J']; J_afm = df[df['J'] > 0]['J']
ax_e.hist([J_fm, J_afm], bins=30, stacked=True, color=['#1f77b4', '#d62728'], label=['FM', 'AFM'])
overall_mean = df['J'].mean(); overall_median = df['J'].median()
stats_text = f'Mean: {overall_mean:.0f} (meV)\nMedian: {overall_median:.0f} (meV)'
ax_e.plot([], [], ' ', label=stats_text); ax_e.set_ylabel('Count')
ax_e.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax_e.set_xlabel('J (meV)')
ax_e.text(0.02, 0.98, panel_labels[4], transform=ax_e.transAxes, fontsize=panel_fontsize, fontweight='bold', va='top', ha='left')
ax_e.legend(loc='center left'); ax_e.grid(False)

# Panel (f): Create 3 stacked subplots
gs_f = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[1, 4], hspace=0)
ax_f1 = fig.add_subplot(gs_f[0, 0])
ax_f2 = fig.add_subplot(gs_f[1, 0], sharex=ax_f1, sharey=ax_f1)
ax_f3 = fig.add_subplot(gs_f[2, 0], sharex=ax_f1, sharey=ax_f1)

ax_f1.scatter(df_sia['N_Fe'], df_sia['eig_max'], s=100, alpha=0.7, color='tab:blue', label='Fe Neighbors')
ax_f1.legend()
ax_f1.text(0.02, 0.85, panel_labels[5], transform=ax_f1.transAxes, fontsize=panel_fontsize, fontweight='bold', va='top', ha='left')

ax_f2.scatter(df_sia['N_Ge'], df_sia['eig_max'], s=100, alpha=0.7, color='tab:orange', label='Ge Neighbors')
ax_f2.legend()
ax_f2.set_ylabel('Maximum absolute eigenvalue of SIA (meV)')

ax_f3.scatter(df_sia['N_total'], df_sia['eig_max'], s=150, alpha=0.9, color='tab:green', label='Total Neighbors')
ax_f3.legend()
ax_f3.set_xlabel('Number of Neighbors')

for ax in [ax_f1, ax_f2, ax_f3]:
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=False))

# Hide redundant tick labels for a cleaner look
plt.setp(ax_f1.get_xticklabels(), visible=False)
plt.setp(ax_f2.get_xticklabels(), visible=False)
plt.setp(ax_a.get_xticklabels(), visible=False)
plt.setp(ax_c.get_xticklabels(), visible=False)
plt.setp(ax_e.get_xticklabels(), visible=False)

# --- 5. Add a Shared, Shorter Colorbar with Specific Ticks ---
gs_cbar = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=gs_main[:, 2])
cbar_ax = fig.add_subplot(gs_cbar[3:6, 0]) 
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label('Magnetic Moment Product (m$_i \cdot$ m$_j$) [$\mu_B^2$]')
cbar_min = df['m_product'].min(); cbar_max = df['m_product'].max()
cbar.set_ticks([cbar_min, 0, cbar_max])
cbar.set_ticklabels([f'{cbar_min:.0f}', '0', f'{cbar_max:.0f}'])

# --- 6. Adjust Layout and Save ---
gs_main.tight_layout(fig)
plt.savefig('Fig3.png', dpi=300, facecolor='white')
plt.show()
