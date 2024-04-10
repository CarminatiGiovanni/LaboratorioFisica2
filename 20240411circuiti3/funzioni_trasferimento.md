# CALCOLO FUNZIONI DI TRASFERIMENTO PER CIRCUITI

## IMPEDENZE:


- CONDENSATORE: $Z_C = \frac{1}{j\omega C}$
- RESISTORE: $Z_R = R$
- INDUTTORE: $Z_L = j\omega L$

## CIRCUITO RC: $Z_{tot} = R + \frac{1}{j\omega C}$

### $H_C(\omega) = \frac{1}{1 + j\omega RC}$
- $|H_C(\omega)| = \frac{1}{\sqrt{1+\omega^2R^2C^2}}$
- $\angle H_C(\omega) = -arctan(\omega RC)$

### $H_R(\omega) = \frac{j\omega RC}{1 + j\omega RC}$
- $|H_C(\omega)| = \frac{\omega RC}{\sqrt{1+\omega^2R^2C^2}}$
- $\angle H_C(\omega) = \frac{\pi}{2}-arctan(\omega RC)$

## CIRCUITO RL: $Z_{tot} = R + R_L + j\omega L$

### $H_L(\omega) = \frac{R_L + j\omega L}{R+R_L+j\omega L}$

- $|H_L(\omega)| = \frac{\sqrt{R_L^2 + \omega^2 L^2}}{\sqrt{(R+R_L)^2 + \omega^2 L^2}}$
- $\angle H_L(\omega) = arctan(\frac{\omega L}{R_L}) - arctan(\frac{\omega L}{R+R_L})$

### $H_R(\omega) = \frac{R}{R+R_L+j\omega L}$

- $|H_R(\omega)| = \frac{R}{\sqrt{(R+R_L)^2 + \omega^2 L^2}}$
- $\angle H_R(\omega) = -arctan(\frac{\omega L}{R+R_L})$
