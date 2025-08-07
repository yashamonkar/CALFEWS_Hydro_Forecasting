from .reservoir_cy cimport Reservoir

cdef class Inputter():
 
  cdef:

    dict __dict__

    str model_mode

    public Reservoir shasta, oroville, folsom, yuba, newmelones, donpedro, exchequer, millerton, sanluisstate, sanluisfederal, sanluis, isabella, success, kaweah, pineflat, trinity

  cdef void run_initialization(self, str plot_key) except *

  cdef void generate_relationships(self, str plot_key) except *

  cdef void generate_relationships_delta(self, str plot_key) except *

  cdef void initialize_reservoirs(self) except *

  cdef void run_routine(self, str flow_input_type, str flow_input_source) except *

  cdef void make_daily_timeseries(self, str flow_input_type, str flow_input_source, int numYears, int start_year, int start_month, int end_month, int first_leap, str plot_key) except *

  cdef void fill_snowpack(self, str plot_key) except *

  cdef tuple whiten_data(self, double[:] data)


