
import matplotlib.pyplot as plt

ACC_random=[67.27, 74.24, 59.43, 76.57, 65.74, 69.22, 61.55, 60.55, 72.75, 81.19]
plt.plot(list(range(len(ACC_random))), ACC_random)
plt.show()
plt.savefig("ACC_indexed_with_OOD_trained_on_random_feature")
ACC_real= [67.27, 74.24, 59.43, 76.57, 65.74, 69.22, 61.55, 60.55, 72.75, 81.19]
plt.plot(list(range(len(ACC_real))), ACC_real)
plt.show()
plt.savefig("ACC indexed with OOD trained on other client feature")