#!/usr/bin/perl -w
use strict;

my %ref;
my %cands;

while (<>) {
  chomp;
  my @fields = split /\t/;
  if (@fields == 2) {  # reference
    $ref{$fields[0]} = $fields[1];
  } elsif (@fields == 3) {  # neighbor
    push @{$cands{$fields[0]}}, $fields[1];
  }
}

my $matched = 0;
while (my ($word, $ref) = each %ref) {
  printf "%s\t%s\t", $word, $ref;
  my @match = grep { $_ eq $ref } @{$cands{$word}};
  printf "%d\n", scalar(@match);
  $matched++ if @match;
}

printf "%d/%d = %.4f\n",
  $matched,
  scalar(keys %ref),
  ($matched / scalar(keys %ref));
