import matplotlib.pyplot as plt

names_fact_q=['Dsc','G=2','G=8']
acc_fact_q=[91.91,93.15,90.32]
scores_fact_q=[0.257091,1.011537,0.276801]

names_dsc_p = ['0.1', '0.2', '0.3']
acc_dsc_p = [90.64, 90.71, 89.63]
scores_dsc_p = [0.221828, 0.18726, 0.155024]

names_dsc_p_dist = ['0.3', '0.5']
acc_dsc_p_dist = [90.83, 90.09]
scores_dsc_p_dist = [0.170323, 0.110921]

names_dist_and_prun = ["G=8, 0.2", "G=2, 0.5", 'G=8, 0.4', 'G=8, 0.5', 'G=8, 0.6']
acc_dist_and_prun = [91.41, 91.87, 90.50, 90.64, 89.96]
scores_dist_and_prun = [0.200057, 0.379571, 0.133548, 0.104438, 0.078058]

names = ['First']
acc = [93.60]
scores = [3.987780]

names_paper = ['Ensemble']
acc_paper = [93.47]
scores_paper = [0.597288]

plt.figure(figsize=(10, 7))
plt.scatter(scores_paper, acc_paper, color='orange', label='Paper')
plt.scatter(scores, acc, color='black', label='First Model')
plt.scatter(scores_fact_q, acc_fact_q, color='green', label='Factorized Models')
plt.scatter(scores_dist_and_prun, acc_dist_and_prun, color='red', label='Distilled and Pruned DSC')
plt.scatter(scores_dsc_p_dist, acc_dsc_p_dist, color='blue', label='Distilled Pruned DSC')
plt.scatter(scores_dsc_p, acc_dsc_p, color='purple', label='Pruned DSC')


for i, name in enumerate(names):
    plt.text(scores[i], acc[i], name, fontsize=9)
for i, name in enumerate(names_paper):
    plt.text(scores_paper[i], acc_paper[i], name, fontsize=9)
for i, name in enumerate(names_dist_and_prun):
    plt.text(scores_dist_and_prun[i], acc_dist_and_prun[i], name, fontsize=9)
for i, name in enumerate(names_fact_q):
    plt.text(scores_fact_q[i], acc_fact_q[i], name, fontsize=9)
for i, name in enumerate(names_dsc_p):
    plt.text(scores_dsc_p[i], acc_dsc_p[i], name, fontsize=9)
for i, name in enumerate(names_dsc_p_dist):
    plt.text(scores_dsc_p_dist[i], acc_dsc_p_dist[i], name, fontsize=9)

#for name in names_dsc_p:
#    if name in names_dsc_p_dist:
#        idx_p = names_dsc_p.index(name)
#        idx_dist = names_dsc_p_dist.index(name)
        
        # Tracer une flèche entre les points correspondants
#        plt.annotate('', 
#                     xy=(scores_dsc_p_dist[idx_dist], acc_dsc_p_dist[idx_dist]), 
#                     xytext=(scores_dsc_p[idx_p], acc_dsc_p[idx_p]),
#                     arrowprops=dict(arrowstyle='->', color='black', lw=2))


# Ajout de la droite pointillée en y=90
plt.axhline(y=90, color='black', linestyle='dashed', linewidth=2)

plt.xlabel('Score')
plt.ylabel('Accuracy (%)')
plt.title('Trade-off Plot')
plt.legend()
plt.grid()
plt.show()
plt.savefig('Lab1/Plots/trade_plot.png')
