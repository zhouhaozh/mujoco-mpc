#pragma once
#include "core/types.h"
namespace sckd {
// Absolute simulation/capture seconds; SI world p/v. One immutable forecast per
// observation version. Consumers must honor false (outside supported domain).
class BallPredictionInterface {
 public:
  virtual ~BallPredictionInterface() = default;
  virtual void Update(const BallObs&) = 0;
  virtual bool initialized() const = 0;
  virtual bool PredictFull(double t, double p[3], double v[3]) const = 0;
  virtual double NextBounceTime() const = 0;
  virtual double t_last() const = 0;
  virtual const double* state() const = 0;
};
}
