#pragma once

void TestIOTime(const int grid_case);

void TestMemoryUsage(void);

void TestRefine(void);

void TestAdvection(const int grid_case);


void TestReduction(void);
void TestReductionSize(void);
void TestDeviceReducer(void);


void TestVelocityExtrapolation(const int grid_case);

void TestParticleBasedRefinement(const int coarse_levels, const int fine_levels, int refine_stride, bool enable_io);

//void TestVorticityCalculation(const int grid_case);


//void TestVelocityJacobian(const int grid_case);
void TestAnalyticalFlowMapAdvectionVariableTime(void);
void TestAnalyticalFlowMapAdvectionFixedTime(void);

//void TestFlowMapAdvection(const int grid_case);

//
// void TestParticleToGridTransfer(const int grid_case);

void TestBinaryFileIO(int grid_case);


//void TestPFMAndNFM(int grid_case);



//void TestLeapFrogAdaptiveFlowMapAdvection(void);

//void TestLeapFrogAdaptiveNFMVorticityAdvection(void);

void TestP2GTransfer(const int grid_case);

void TestG2PAdvectionEffeciency(int grid_case);
void TestParticleToGridTransferEfficiency(int grid_case);

void TestCertainAdvection(int num_tiles);