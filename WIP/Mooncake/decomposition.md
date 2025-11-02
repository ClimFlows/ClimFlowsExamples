Sik,mik => Elemental(/, 2=>1) => sk
uek, mik => radius^-2*centered_flux => Uek 
Ue, sk => centered_flux => sUe
Uek => (-1)*div => dmik
sUe => (-1)*div => dSik

uek => (1/2)*norm2 => Kik
mik => cumsum_down(ptop) => pil
pil => half_to_full => pik
pik, sik => Elemental(thermo, 2=>3) => hik, vik, pik  ( h-s*h_s , h_p, h_s ; h_p-s*h_sp, -s*h_ss, h_ps, h_pp, h_sp, h_ss)
vik, mik => Jac*Elemental(*, 2=>1) => dPhi
dPhi => cumsum_up(Phis) => Phi_l
Phil_l => half_to_full => Phi_k
Phik, Kik, hik => Elemental(+, 3=>1) => Bik
uek => curl => zvk
mik => avg_iv => mvk
zvk, mvk => Elemental(2=>1, PV, fv) => qvk
qvk => avg_ve => qek
qek, Uek => trisk_K => due_curl_ek
Bik, sik, pik => bracket_grad => due_grad_ek

- horizontal stencils
  centered_flux i,e => e
  norm2 e => i
  curl e => v
  div e => i
  trisk_K e,e => e
  bracket_grad i,i,i => e
  avg_iv i=>v
  avg_ve v=>e

-vertical stencils
  half_to_full o cumsum_down
  half_to_full o cumsum_up

- Elemental
   /, 2=>1
   *, 2=>1
   +, 3=>1
   +, 2=>1
   PV, 2=>1
   thermo, 2=>3

