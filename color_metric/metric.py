# AI-assisted experimental code; full human and mathematical review is not established.
"""Interpolate chromaticity metrics and sample polynomial Killing equations.

Measurements are (x,y,a,b,angle_deg) discrimination ellipses. MetricField returns
a symmetric 2-by-2 matrix; its default logarithmic interpolation preserves
positive definiteness. killing_matrix maps polynomial vector coefficients to
sampled metric-preservation residuals, not a proof about all smooth fields.
"""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import RBFInterpolator
from matplotlib.path import Path as Polygon


def macadam_data():
    """Return 25 rows of (x,y,a,b,angle_deg) discrimination-ellipse measurements.

    a and b are positive semiaxis lengths in chromaticity coordinates; angle
    is counterclockwise in degrees. The bundled data/NOTICE gives provenance.
    """
    return np.asarray(json.loads((Path(__file__).parent/'data/macadam.json').read_text()))


def ellipse_metric(a,b,angle_deg):
    """Metric whose unit ellipse has axes a, b and counterclockwise angle."""
    if a <= 0 or b <= 0 or not np.isfinite([a,b,angle_deg]).all():
        raise ValueError("Ellipse axes must be positive and inputs finite")
    c,s=np.cos(np.radians(angle_deg)),np.sin(np.radians(angle_deg))
    return np.array([[(c/a)**2+(s/b)**2,c*s*(1/a**2-1/b**2)],
                     [c*s*(1/a**2-1/b**2),(s/a)**2+(c/b)**2]])


def _symmetric_function(matrix, function):
    values,vectors=np.linalg.eigh(matrix)
    return (vectors*function(values))@vectors.T


class MetricField:
    """Thin-plate radial-basis interpolation of 2-by-2 discrimination metrics.

    log-euclidean interpolates matrix logarithms and exponentiates the result,
    preserving positive definiteness. article-component interpolates the three
    independent matrix entries directly and can produce indefinite matrices.
    Calling the field at (x,y) returns one symmetric matrix. smoothing is the
    radial-basis regularization parameter, with default 1.0.
    """
    def __init__(self,mode="log-euclidean",smoothing=1.0):
        if mode not in ("log-euclidean","article-component"):
            raise ValueError("Unknown interpolation mode")
        self.mode=mode
        data=macadam_data()
        matrices=np.array([ellipse_metric(a,b,angle) for x,y,a,b,angle in data])
        if mode=="log-euclidean":
            matrices=np.array([_symmetric_function(g,np.log) for g in matrices])
        components=matrices[:,[0,0,1],[0,1,1]]
        self.interpolator=RBFInterpolator(data[:,:2],components,kernel="thin_plate_spline",smoothing=smoothing)

    def __call__(self,x,y):
        a,b,c=self.interpolator([[x,y]])[0]
        g=np.array([[a,b],[b,c]])
        if self.mode=="log-euclidean":
            g=_symmetric_function(g,np.exp)
        return g


def gamut_grid():
    """Return a rectangle grid clipped to the bundled spectral-locus polygon.

    Sample 20 x values from 0.10 to 0.65 and 20 y values from 0.08 to 0.65.
    Keep the 332 candidate points inside the polygon encoded in this function.
    """
    x=np.array([0.1741, 0.174, 0.1714, 0.1644, 0.1566, 0.144, 0.1241, 0.0913, 0.0633, 0.0235, 0.0082, 0.0139, 0.0743, 0.1547, 0.2296, 0.295, 0.3616, 0.4294, 0.5028, 0.5706, 0.6256, 0.6658, 0.6915, 0.7079, 0.719, 0.726, 0.73, 0.732, 0.7334, 0.7344, 0.7347, 0.7347, 0.7347])
    y=np.array([0.005, 0.005, 0.0065, 0.0109, 0.0177, 0.0297, 0.0578, 0.1327, 0.265, 0.4073, 0.5384, 0.6548, 0.7243, 0.7514, 0.7543, 0.7449, 0.73, 0.7106, 0.6858, 0.6562, 0.6229, 0.5858, 0.5475, 0.5123, 0.4813, 0.4562, 0.4353, 0.4188, 0.4044, 0.3935, 0.3872, 0.3848, 0.383])
    path=Polygon(np.column_stack([np.append(x,x[0]),np.append(y,y[0])]))
    return np.array([(a,b) for a in np.linspace(.10,.65,20) for b in np.linspace(.08,.65,20) if path.contains_point((a,b))])


def poly_basis(x,y,degree=3):
    """Monomials ordered by ascending x power, then ascending y power."""
    return np.array([x**i*y**j for i in range(degree+1) for j in range(degree+1-i)])


def killing_matrix(metric,points,degree=3,h=.005):
    """Return sampled coefficients of the metric-preservation equation L_X g = 0.

    metric(x,y) returns a 2-by-2 matrix; points has shape (N,2). Each component
    of X is a polynomial of total degree <= degree, with monomials ordered by
    poly_basis. Columns list the first component's coefficients, then the second.
    Rows are (00,01,11) at each point. Derivatives use central differences with
    step h, giving shape (3*N, (degree+1)*(degree+2)). A finite sampled nullspace
    is evidence only for this ansatz and sampling, not all smooth vector fields.
    """
    if not isinstance(degree,int) or degree<0 or not np.isfinite(h) or h<=0:
        raise ValueError("Require integer degree >= 0 and finite h > 0")
    n=len(poly_basis(0,0,degree));rows=[]
    for x,y in points:
        g=metric(x,y)
        dg=[(metric(x+h,y)-metric(x-h,y))/(2*h), (metric(x,y+h)-metric(x,y-h))/(2*h)]
        phi=poly_basis(x,y,degree)
        dphi=[(poly_basis(x+h,y,degree)-poly_basis(x-h,y,degree))/(2*h),
              (poly_basis(x,y+h,degree)-poly_basis(x,y-h,degree))/(2*h)]
        for i,j in [(0,0),(0,1),(1,1)]:
            row=np.zeros(2*n)
            for k in (0,1):
                row[k*n:(k+1)*n]=phi*dg[k][i,j]+g[k,j]*dphi[i]+g[i,k]*dphi[j]
            rows.append(row)
    return np.array(rows)


def evaluate(mode="log-euclidean",rtol=1e-10):
    metric=MetricField(mode);points=gamut_grid()
    eigenvalues=np.linalg.eigvalsh([metric(*p) for p in points])
    A=killing_matrix(metric,points)
    singular_values=np.linalg.svd(A,compute_uv=False)
    rank=int(np.sum(singular_values>rtol*singular_values[0]))
    return dict(mode=mode,grid_points=len(points),shape=list(A.shape),rank=rank,
                rank_relative_tolerance=rtol,singular_values=singular_values.tolist(),
                smallest_relative_singular_value=float(singular_values[-1]/singular_values[0]),
                nonpositive_metric_points=int((eigenvalues[:,0]<=0).sum()),
                minimum_metric_eigenvalue=float(eigenvalues.min()))
