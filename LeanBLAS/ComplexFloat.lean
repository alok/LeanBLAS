

namespace BLAS


/-- Complex number with double-precision real and imaginary parts. -/
structure ComplexFloat where
  re : Float
  im : Float
  deriving Inhabited

instance : ToString ComplexFloat where
  toString c := toString c.re ++ " + " ++ toString c.im ++ "ⅈ"

instance : BEq ComplexFloat where
  beq a b := a.re == b.re && a.im == b.im

instance : Add ComplexFloat where
  add a b := ⟨a.re + b.re, a.im + b.im⟩

instance : Sub ComplexFloat where
  sub a b := ⟨a.re - b.re, a.im - b.im⟩

instance : Mul ComplexFloat where
  mul a b := ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩

instance : Div ComplexFloat where
  div a b :=
    let d := b.re * b.re + b.im * b.im
    ⟨(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d⟩

instance : HDiv ComplexFloat Float ComplexFloat where
  hDiv a b := ⟨a.re / b, a.im / b⟩

instance : HMul ComplexFloat Float ComplexFloat where
  hMul a b := ⟨a.re * b, a.im * b⟩

instance : HMul Float ComplexFloat ComplexFloat where
  hMul a b := ⟨a * b.re, a * b.im⟩

instance : Neg ComplexFloat where
  neg a := ⟨-a.re, -a.im⟩

def ComplexFloat.zero : ComplexFloat := ⟨0, 0⟩

def ComplexFloat.one : ComplexFloat := ⟨1, 0⟩

def ComplexFloat.I : ComplexFloat := ⟨0, 1⟩

def ComplexFloat.abs (a : ComplexFloat) : Float := Float.sqrt (a.re * a.re + a.im * a.im)

def ComplexFloat.conj (a : ComplexFloat) : ComplexFloat := ⟨a.re, -a.im⟩

def ComplexFloat.exp (a : ComplexFloat) : ComplexFloat :=
  let e := Float.exp a.re
  ⟨e * Float.cos a.im, e * Float.sin a.im⟩

def ComplexFloat.log (a : ComplexFloat) : ComplexFloat :=
  let r := ComplexFloat.abs a
  let θ := Float.atan2 a.im a.re
  ⟨Float.log r, θ⟩

def ComplexFloat.pow (a b : ComplexFloat) : ComplexFloat :=
  ComplexFloat.exp (b * ComplexFloat.log a)

def ComplexFloat.sqrt (a : ComplexFloat) : ComplexFloat :=
  let r := ComplexFloat.abs a
  let θ := Float.atan2 a.im a.re
  let r' := Float.sqrt r
  let θ' := θ / 2
  ⟨r' * Float.cos θ', r' * Float.sin θ'⟩

def ComplexFloat.cbrt (a : ComplexFloat) : ComplexFloat :=
  let r := ComplexFloat.abs a
  let θ := Float.atan2 a.im a.re
  let r' := Float.cbrt r
  let θ' := θ / 3
  ⟨r' * Float.cos θ', r' * Float.sin θ'⟩

def ComplexFloat.sin (a : ComplexFloat) : ComplexFloat :=
  let re := a.re
  let im := a.im
  ⟨Float.sin re * Float.cosh im, Float.cos re * Float.sinh im⟩

def ComplexFloat.cos (a : ComplexFloat) : ComplexFloat :=
  let re := a.re
  let im := a.im
  ⟨Float.cos re * Float.cosh im, -Float.sin re * Float.sinh im⟩

def ComplexFloat.tan (a : ComplexFloat) : ComplexFloat :=
  ComplexFloat.sin a / ComplexFloat.cos a

def ComplexFloat.asin (a : ComplexFloat) : ComplexFloat :=
  -ComplexFloat.log (ComplexFloat.one * ComplexFloat.I - a * ComplexFloat.I) * ComplexFloat.I

def ComplexFloat.acos (a : ComplexFloat) : ComplexFloat :=
  -ComplexFloat.log (a + ComplexFloat.I * ComplexFloat.sqrt (ComplexFloat.one - a * a)) * ComplexFloat.I

def ComplexFloat.atan (a : ComplexFloat) : ComplexFloat :=
  (ComplexFloat.log (ComplexFloat.one - a * ComplexFloat.I) - ComplexFloat.log (ComplexFloat.one + a * ComplexFloat.I)) * ComplexFloat.I / (2.0 : Float)

def ComplexFloat.sinh (a : ComplexFloat) : ComplexFloat :=
  let re := a.re
  let im := a.im
  ⟨Float.sinh re * Float.cos im, Float.cosh re * Float.sin im⟩

def ComplexFloat.cosh (a : ComplexFloat) : ComplexFloat :=
  let re := a.re
  let im := a.im
  ⟨Float.cosh re * Float.cos im, Float.sinh re * Float.sin im⟩

def ComplexFloat.tanh (a : ComplexFloat) : ComplexFloat :=
  ComplexFloat.sinh a / ComplexFloat.cosh a

def ComplexFloat.asinh (a : ComplexFloat) : ComplexFloat :=
  ComplexFloat.log (a + ComplexFloat.sqrt (a * a + ComplexFloat.one))

def ComplexFloat.acosh (a : ComplexFloat) : ComplexFloat :=
  ComplexFloat.log (a + ComplexFloat.sqrt (a * a - ComplexFloat.one))

def ComplexFloat.atanh (a : ComplexFloat) : ComplexFloat :=
  (ComplexFloat.log (ComplexFloat.one + a) - ComplexFloat.log (ComplexFloat.one - a)) / (2.0:Float)


def ComplexFloat.floor (a : ComplexFloat) : ComplexFloat := ⟨Float.floor a.re, Float.floor a.im⟩

def ComplexFloat.ceil (a : ComplexFloat) : ComplexFloat := ⟨Float.ceil a.re, Float.ceil a.im⟩

def ComplexFloat.round (a : ComplexFloat) : ComplexFloat := ⟨Float.round a.re, Float.round a.im⟩

def ComplexFloat.isFinite (a : ComplexFloat) : Bool := a.re.isFinite && a.im.isFinite

def ComplexFloat.isNaN (a : ComplexFloat) : Bool := a.re.isNaN || a.im.isNaN

def ComplexFloat.isInf (a : ComplexFloat) : Bool := a.re.isInf || a.im.isInf


----------------------------------------------------------------------------------------------------


structure ComplexFloatArray where
  data : FloatArray
  deriving Inhabited

def ComplexFloatArray.size (a : ComplexFloatArray) : Nat := a.data.size / 2

def ComplexFloatArray.get! (a : ComplexFloatArray) (i : Nat) : ComplexFloat :=
  ⟨a.data.get! (2*i), a.data.get! (2*i+1)⟩

def ComplexFloatArray.get? (a : ComplexFloatArray) (i : Nat) : Option ComplexFloat :=
  if i < a.size then some (a.get! i)
  else none

def ComplexFloatArray.get (a : ComplexFloatArray) (i : Fin a.size) : ComplexFloat :=
  a.get! i.val

def ComplexFloatArray.set! (a : ComplexFloatArray) (i : Nat) (c : ComplexFloat) : ComplexFloatArray :=
  ⟨a.data.set! (2*i) c.re |>.set! (2*i+1) c.im⟩

def ComplexFloatArray.set (a : ComplexFloatArray) (i : Fin a.size) (c : ComplexFloat) : ComplexFloatArray :=
  a.set! i.val c

def ComplexFloatArray.mkEmpty : ComplexFloatArray := ⟨FloatArray.empty⟩

def ComplexFloatArray.push (a : ComplexFloatArray) (c : ComplexFloat) : ComplexFloatArray :=
  ⟨(a.data.push c.re).push c.im⟩

def ComplexFloatArray.ofArray (xs : Array ComplexFloat) : ComplexFloatArray := Id.run do
  let mut data := FloatArray.empty
  for c in xs do
    data := data.push c.re
    data := data.push c.im
  return ⟨data⟩


instance : ToString ComplexFloatArray where
  toString a := Id.run do
    let mut r := "#[" ++ (toString (a.get! 0))
    for i in [1:a.size] do
      r := r ++ ", " ++ toString (a.get! i)
    r := r ++ "]"
    pure r
