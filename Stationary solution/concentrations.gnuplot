set xlabel 'Dimensionless position'
set ylabel 'Dimensionless concentration'

set key bottom right

set term png
set output 'concentrations.png'

plot 'Computed_2_grid100.csv' u 1:3 w lines title 'Exact' smooth unique, 'Computed_1_grid10.csv' u 1:2 w lines title 'Degree = 1; Grid = 10' smooth unique, 'Computed_1_grid100.csv' u 1:2 w lines title 'Degree = 1; Grid = 100' smooth unique, 'Computed_2_grid10.csv' u 1:2 w lines title 'Degree = 2; Grid = 10' smooth unique, 'Computed_2_grid100.csv' u 1:2 w lines title 'Degree = 2; Grid = 100' smooth unique
