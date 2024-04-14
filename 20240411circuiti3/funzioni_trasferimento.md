# CALCOLO FUNZIONI DI TRASFERIMENTO PER CIRCUITI

## IMPEDENZE

- CONDENSATORE: $Z_C = \frac{1}{j\omega C}$
- RESISTORE: $Z_R = R$
- INDUTTORE: $Z_L = j\omega L$

## DEFINIZIONE FUNZIONE DI TRASFERIMENTO

$V_{out} = V_{in} H(\omega)$ la tensione misurata su un componente ($V_{out}$)dipende dalla tensione in ingresso nel circuito $V_{in}$ moltiplicato per una funzione dipendente dalla frequenza detta FUNZIONE DI TRASFERIMENTO

### Per un generatore di onde sinusoidali

$V_{in} = V_G e^{j\omega t}$

$V_{out} = H(\omega)V_{in}$ dove $H(\omega) \in \Complex$

$H(\omega) = \rho e^{j\Phi}$ ha una fase e un modulo costanti nel tempo

$\Rightarrow V_{out} = V_G\cdot\rho\cdot e^{i\omega t}\cdot e^{i\Phi}$

$\Rightarrow V_{out} = V_G\cdot|H(\omega)|\cdot e^{i\omega t}\cdot e^{i\angle H(\omega)}$

## CIRCUITO RC: $Z_{tot} = R + \frac{1}{j\omega C}$

### $H_C(\omega) = \frac{1}{1 + j\omega RC}$

- $|H_C(\omega)| = \frac{1}{\sqrt{1+\omega^2R^2C^2}}$
- $\angle H_C(\omega) = -arctan(\omega RC)$

### $H_R(\omega) = \frac{j\omega RC}{1 + j\omega RC}$

- $|H_R(\omega)| = \frac{\omega RC}{\sqrt{1+\omega^2R^2C^2}}$
- $\angle H_R(\omega) = \frac{\pi}{2}-arctan(\omega RC)$

## CIRCUITO RL: $Z_{tot} = R + R_L + j\omega L$

### $H_L(\omega) = \frac{R_L + j\omega L}{R+R_L+j\omega L}$

- $|H_L(\omega)| = \frac{\sqrt{R_L^2 + \omega^2 L^2}}{\sqrt{(R+R_L)^2 + \omega^2 L^2}}$
- $\angle H_L(\omega) = arctan(\frac{\omega L}{R_L}) - arctan(\frac{\omega L}{R+R_L})$

### $H_R(\omega) = \frac{R}{R+R_L+j\omega L}$

- $|H_R(\omega)| = \frac{R}{\sqrt{(R+R_L)^2 + \omega^2 L^2}}$
- $\angle H_R(\omega) = -arctan(\frac{\omega L}{R+R_L})$

## CIRCUITO RLC: $Z_{tot} = R + R_L + j\omega L + \frac{1}{j \omega C}$

$Z_{tot} = \frac{j\omega C}{(R + R_L)\omega Cj + 1 - \omega^2 LC}$

### $H_R(\omega) = \frac{j\omega CR}{(R + R_L)\omega Cj + 1 - \omega^2 LC}$

- $|H_R(\omega)| = \frac{\omega CR}{\sqrt{(1-\omega^2LC)^2 + \omega^2C^2 (R+R_L)^2}}$

- $\angle H_R(\omega) = \frac{\pi}{2} -arctan(\frac{\omega C(R+R_L)}{1-\omega^2LC})$

### $H_C(\omega) = \frac{1}{(R + R_L)\omega Cj + 1 - \omega^2 LC}$

- $|H_C(\omega)| = \frac{1}{\sqrt{(1-\omega^2LC)^2 + \omega^2C^2 (R+R_L)^2}}$
- $\angle H_C(\omega) = -arctan(\frac{\omega C(R+R_L)}{1-\omega^2LC})$

### $H_L(\omega) = \frac{\omega CR_Lj - \omega^2LC}{(R + R_L)\omega Cj + 1 - \omega^2 LC}$

- $|H_L(\omega)| = \frac{\sqrt{\omega^2C^2R_L^2+\omega^4L^2C^2}}{\sqrt{(1-\omega^2LC)^2 + \omega^2C^2 (R+R_L)^2}}$
- $\angle H_L(\omega) = -arctan(\frac{R_L}{\omega L}) - arctan(\frac{\omega C(R+R_L)}{1-\omega^2LC})$
