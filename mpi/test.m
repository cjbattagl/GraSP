  A=load('mygraphtup.mat');
  B=load('mygraphcsr.mat');
  C=load('mygraphperm.mat');
  D=load('mygraphpcsr.mat');
  A = spconvert(A);
  A(max(size(A)),max(size(A))) = 1;
  B = spconvert(B);
  C = spconvert(C);
  D = spconvert(D);
  hold on;

  %A(4096,4096)=1;   B(4096,4096)=1;   C(4096,4096)=1;   D(4096,4096)=1;

  if (1==1)
    subplot(5,1,1)
    spy(A,'.');
    hold on;
    subplot(5,1,2)
    spy(B,'.')
    subplot(5,1,3)
    spy(C,'.')
    subplot(5,1,4)
    spy(D,'.')
  end
  if (1==1)
  [nnz(A) nnz(B) nnz(C) nnz(D)]
  [size(A); size(B); size(C); size(D)]
  b=load('parts.mat');
  nparts = max(b(:,2));
  P = [];
  idxlist = zeros(nparts,2); %contains indices of partition matrix
  psizes = zeros(nparts,1);
  N = size(B,1);
  B(N,N)=1;
  %assert(N==size(a,1));
  rem = 1:N;
  for i=1:nparts
    [t idx] = find(b(:,2)==i);
    verts = b(t,1);
    lo = length(P) + 1;
    rem(verts) = -1;
    P = [P; verts];
    hi = lo + length(t) - 1;
    idxlist(i,:) = [lo hi];
    psizes(i) = length(t);
  end
  t = find(rem==-1);
  rem(t) = [];
  rem = rem';
  %C = B(P,P);
  E = B([rem; P],[rem; P]);

  % Sanity checks
  assert(length(unique(P))==length(P));
  %assert(length(P)==N);
  %assert(nnz(A)==nnz(B));
  subplot(5,1,5)
  spy(E,'.')
  end
