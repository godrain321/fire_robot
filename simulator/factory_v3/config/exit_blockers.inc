! Human-edited scenario obstacle. This file is intentionally separate from
! generated/obstacles.inc and generated/exits.inc so the base map is unchanged.

&SURF ID='EXIT1_BLOCKER', COLOR='ORANGE' /

! Fallen storage rack across the interior side of EXIT1.
! The rack spans the exit corridor and is low enough to remain visibly
! different from the 3 m walls while still being non-traversable.
&OBST ID='EXIT1_FALLEN_STORAGE_RACK',
      XB=7.800,10.800,17.800,18.000,0.000,1.400,
      SURF_ID='EXIT1_BLOCKER' /
